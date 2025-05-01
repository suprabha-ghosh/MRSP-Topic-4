import os
import librosa
import openl3
import torch
import numpy as np
import soundfile as sf
from torchvggish import vggish, vggish_input

# Directory paths
INPUT_DIR = "test_audio"
OPENL3_OUTPUT_DIR = "openl3_embeddings"
VGGISH_OUTPUT_DIR = "vggish_embeddings"

def setup_directories():
    """Create output directories if they don't exist"""
    for directory in [INPUT_DIR, OPENL3_OUTPUT_DIR, VGGISH_OUTPUT_DIR]:
        os.makedirs(directory, exist_ok=True)

def create_degraded_audio(input_file, output_file, noise_level=0.005):
    """
    Create a degraded version of the input audio by adding noise
    Args:
        input_file: path to input audio file
        output_file: path to save degraded audio
        noise_level: amount of noise to add
    """
    print(f"\nCreating degraded version with noise level {noise_level}...")
    
    # Load the audio file
    audio, sr = librosa.load(input_file, sr=None)
    
    # Add noise
    noise = np.random.normal(0, noise_level, len(audio))
    degraded_audio = audio + noise
    
    # Ensure the audio is normalized
    degraded_audio = librosa.util.normalize(degraded_audio)
    
    # Save the degraded audio
    sf.write(output_file, degraded_audio, sr)
    
    print(f"Created degraded version: {output_file}")

def extract_embeddings(audio_path, output_filename):
    """Extract both OpenL3 and VGGish embeddings"""
    print(f"\nProcessing: {audio_path}")
    
    try:
        # Extract OpenL3 embeddings
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        emb_openl3, _ = openl3.get_audio_embedding(audio, sr, 
                                                  input_repr="mel256", 
                                                  content_type="music", 
                                                  embedding_size=512)
        
        # Save OpenL3 embeddings
        openl3_path = os.path.join(OPENL3_OUTPUT_DIR, f"{output_filename}.npy")
        np.save(openl3_path, emb_openl3)
        print(f"Saved OpenL3 embeddings: {openl3_path}")
        
        # Extract VGGish embeddings
        model = vggish()
        model.eval()
        
        # Ensure audio is at 16kHz for VGGish
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Convert to the format expected by VGGish
        examples = vggish_input.waveform_to_examples(audio, sr)
        
        # Ensure examples is a numpy array
        if torch.is_tensor(examples):
            examples = examples.detach().numpy()
            
        # Convert to torch tensor for model input
        examples = torch.FloatTensor(examples)
        
        with torch.no_grad():
            emb_vggish = model(examples)
            # Convert tensor to numpy array before saving
            emb_vggish = emb_vggish.detach().numpy()
        
        # Save VGGish embeddings
        vggish_path = os.path.join(VGGISH_OUTPUT_DIR, f"{output_filename}.npy")
        np.save(vggish_path, emb_vggish)
        print(f"Saved VGGish embeddings: {vggish_path}")
        
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        raise

def main():
    """Main function to process audio files"""
    try:
        # Setup directories
        setup_directories()
        
        # Define noise levels for degraded versions
        noise_levels = [0.001, 0.002, 0.005, 0.01, 0.02]
        
        # Process original (clean) audio
        original_audio = "test_audio/test_audio.wav"
        extract_embeddings(original_audio, "test_audio_clean")
        
        # Create and process degraded versions
        for i, noise_level in enumerate(noise_levels):
            # Create degraded version
            degraded_filename = f"test_audio_degraded_{i+1}.wav"
            degraded_path = os.path.join(INPUT_DIR, degraded_filename)
            create_degraded_audio(original_audio, degraded_path, noise_level)
            
            # Extract embeddings for degraded version
            output_filename = f"test_audio_degraded_{i+1}"
            extract_embeddings(degraded_path, output_filename)
        
        print("\nEmbedding extraction completed successfully!")
        
        # Print summary of created files
        print("\nCreated files:")
        print("\nAudio files:")
        for f in sorted(os.listdir(INPUT_DIR)):
            print(f"  {f}")
            
        print("\nOpenL3 embeddings:")
        for f in sorted(os.listdir(OPENL3_OUTPUT_DIR)):
            print(f"  {f}")
            
        print("\nVGGish embeddings:")
        for f in sorted(os.listdir(VGGISH_OUTPUT_DIR)):
            print(f"  {f}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main()
