import os
import librosa
import openl3
import torch
import numpy as np
import soundfile as sf
from torchvggish import vggish, vggish_input
from scipy import signal
import argparse
from tqdm import tqdm

# Default directory paths
DEFAULT_INPUT_DIR = "test_audio"
DEFAULT_OPENL3_DIR = "openl3_embeddings"
DEFAULT_VGGISH_DIR = "vggish_embeddings"

def setup_directories(dirs):
    """Create directories if they don't exist"""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

def create_noisy_audio(audio, noise_value, seed=None):
    """Create a degraded version of the audio by adding noise"""
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(0, noise_value, len(audio))
    degraded = audio + noise
    return librosa.util.normalize(degraded)

def create_bitrate_audio(audio, sr, factor):
    """Simulate bitrate reduction by downsampling and upsampling"""
    downsampled = librosa.resample(audio, orig_sr=sr, target_sr=sr // factor)
    upsampled = librosa.resample(downsampled, orig_sr=sr // factor, target_sr=sr)
    upsampled = librosa.util.fix_length(upsampled, size=len(audio))
    return librosa.util.normalize(upsampled)

def create_compressed_audio(audio, bits, quality="low"):
    """Simulate compression artifacts by quantizing the audio"""
    max_val = 2**(bits - 1) - 1
    step = 1.0 / max_val
    quantized = np.round(audio / step) * step
    quantized = np.clip(quantized, -1.0, 1.0)
    
    # Apply different filtering based on quality level
    if quality == "extreme":
        # More aggressive filtering for extreme compression
        b, a = signal.butter(6, 0.5, btype='low', analog=False)
        quantized = signal.filtfilt(b, a, quantized)
    elif quality == "low":
        # Standard filtering for low quality
        b, a = signal.butter(4, 0.7, btype='low', analog=False)
        quantized = signal.filtfilt(b, a, quantized)
    # No filtering for high quality
    
    return librosa.util.normalize(quantized)

def extract_embeddings(audio, sr, output_filename, openl3_dir, vggish_dir, skip_existing=False):
    """Extract OpenL3 and VGGish embeddings from audio"""
    openl3_path = os.path.join(openl3_dir, f"{output_filename}.npy")
    vggish_path = os.path.join(vggish_dir, f"{output_filename}.npy")
    
    # Skip if both embeddings already exist
    if skip_existing and os.path.exists(openl3_path) and os.path.exists(vggish_path):
        return openl3_path, vggish_path
    
    # OpenL3
    emb_openl3, _ = openl3.get_audio_embedding(audio, sr, input_repr="mel256", content_type="music", embedding_size=512)
    np.save(openl3_path, emb_openl3)
    
    # VGGish
    model = vggish()
    model.eval()

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    examples = vggish_input.waveform_to_examples(audio, sr)
    if torch.is_tensor(examples):
        examples = examples.detach().numpy()
    input_tensor = torch.FloatTensor(examples)

    with torch.no_grad():
        emb_vggish = model(input_tensor).detach().numpy()

    np.save(vggish_path, emb_vggish)
    
    return openl3_path, vggish_path

def process_audio_file(audio_path, input_dir, openl3_dir, vggish_dir, apply_degradations=True, skip_existing=False):
    """Process a single audio file and create degraded versions"""
    try:
        # Get base filename without extension
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        print(f"\nProcessing: {audio_path} ({len(audio)/sr:.2f} sec, {sr} Hz)")
        
        # Process clean version
        clean_output_name = f"{filename}_clean"
        openl3_path, vggish_path = extract_embeddings(audio, sr, clean_output_name, openl3_dir, vggish_dir, skip_existing)
        print(f"Saved clean embeddings: OpenL3 ({os.path.getsize(openl3_path)//1024} KB), VGGish ({os.path.getsize(vggish_path)//1024} KB)")
        
        if not apply_degradations:
            return True
            
        # Define degradation types
        degradations = [
            # Noise degradations
            {"type": "noise", "param": 0.002, "suffix": "noise_low"},      # Low amount of noise (best quality)
            {"type": "noise", "param": 0.01, "suffix": "noise_high"},      # High amount of noise (worse quality)
            {"type": "noise", "param": 0.02, "suffix": "noise_extreme"},   # Extreme amount of noise (worst quality)

            # Bitrate degradations
            {"type": "bitrate", "param": 2, "suffix": "bitrate_low"},      # Low degradation (best quality)
            {"type": "bitrate", "param": 4, "suffix": "bitrate_high"},     # High degradation (worse quality)
            {"type": "bitrate", "param": 6, "suffix": "bitrate_extreme"},  # Extreme degradation (worst quality)

            # Compression degradations
            {"type": "compression", "param": 12, "suffix": "compression_low"},      # Low degradation (best quality)
            {"type": "compression", "param": 8, "suffix": "compression_high"},      # High degradation (worse quality)
            {"type": "compression", "param": 4, "suffix": "compression_extreme"}    # Extreme degradation (worst quality)
        ]
        
        # Process each degradation type
        for d in tqdm(degradations, desc="Creating degraded versions"):
            suffix = d['suffix']
            output_audio_name = f"{filename}_{suffix}"
            output_audio_path = os.path.join(input_dir, f"{output_audio_name}.wav")
            
            # Skip if output audio and embeddings already exist
            if skip_existing:
                openl3_exists = os.path.exists(os.path.join(openl3_dir, f"{output_audio_name}.npy"))
                vggish_exists = os.path.exists(os.path.join(vggish_dir, f"{output_audio_name}.npy"))
                if os.path.exists(output_audio_path) and openl3_exists and vggish_exists:
                    print(f"Skipping existing: {output_audio_name}")
                    continue
            
            # Create degraded version
            if d["type"] == "noise":
                degraded = create_noisy_audio(audio, d["param"], seed=42)  # Fixed seed for reproducibility
            elif d["type"] == "bitrate":
                degraded = create_bitrate_audio(audio, sr, d["param"])
            elif d["type"] == "compression":
                quality = "low" if "low" in suffix else "extreme" if "extreme" in suffix else "high"
                degraded = create_compressed_audio(audio, d["param"], quality)
            else:
                continue
                
            # Save degraded audio
            sf.write(output_audio_path, degraded, sr)
            
            # Extract embeddings
            openl3_path, vggish_path = extract_embeddings(degraded, sr, output_audio_name, openl3_dir, vggish_dir, skip_existing)
            
        return True
        
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def batch_process(input_dir, openl3_dir, vggish_dir, clean_only=False, skip_existing=False):
    """Process all audio files in the input directory"""
    # Get all audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    audio_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if os.path.splitext(f.lower())[1] in audio_extensions
    ]
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
        
    print(f"Found {len(audio_files)} audio files to process")
    
    # Process each file
    successful = 0
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        if process_audio_file(audio_file, input_dir, openl3_dir, vggish_dir, not clean_only, skip_existing):
            successful += 1
            
    print(f"\n✅ Processed {successful}/{len(audio_files)} files successfully!")

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Extract audio embeddings in batch")
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR, help="Input directory containing audio files")
    parser.add_argument("--clean-only", action="store_true", help="Only extract embeddings for clean audio (no degradations)")
    parser.add_argument("--openl3-dir", default=DEFAULT_OPENL3_DIR, help="Output directory for OpenL3 embeddings")
    parser.add_argument("--vggish-dir", default=DEFAULT_VGGISH_DIR, help="Output directory for VGGish embeddings")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that already have embeddings")
    
    args = parser.parse_args()
    
    input_dir = args.input
    openl3_dir = args.openl3_dir
    vggish_dir = args.vggish_dir
    
    try:
        # Setup directories
        setup_directories([input_dir, openl3_dir, vggish_dir])
        
        # Process files in batch
        batch_process(input_dir, openl3_dir, vggish_dir, args.clean_only, args.skip_existing)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
