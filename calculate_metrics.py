import os
import numpy as np
import librosa
import pandas as pd
from scipy.spatial.distance import cosine
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm
import argparse

# Default directory paths
DEFAULT_AUDIO_DIR = "test_audio"
DEFAULT_OPENL3_DIR = "openl3_embeddings"
DEFAULT_VGGISH_DIR = "vggish_embeddings"
DEFAULT_RESULTS_DIR = "results"

# Updated degradation types
DEGRADATIONS = [
    # Noise degradations
    {"suffix": "noise_low", "type": "noise", "quality": "low", "display": "Noise (Low)"},        # Low amount, best quality
    {"suffix": "noise_high", "type": "noise", "quality": "high", "display": "Noise (High)"},     # High amount, worse quality
    {"suffix": "noise_extreme", "type": "noise", "quality": "extreme", "display": "Noise (Extreme)"}, # Extreme, worst quality

    # Bitrate degradations
    {"suffix": "bitrate_low", "type": "bitrate", "quality": "low", "display": "Bitrate (Low)"},        # Low degradation, best quality
    {"suffix": "bitrate_high", "type": "bitrate", "quality": "high", "display": "Bitrate (High)"},     # High degradation, worse quality
    {"suffix": "bitrate_extreme", "type": "bitrate", "quality": "extreme", "display": "Bitrate (Extreme)"}, # Extreme, worst quality

    # Compression degradations
    {"suffix": "compression_low", "type": "compression", "quality": "low", "display": "Compression (Low)"},        # Low degradation, best quality
    {"suffix": "compression_high", "type": "compression", "quality": "high", "display": "Compression (High)"},     # High degradation, worse quality
    {"suffix": "compression_extreme", "type": "compression", "quality": "extreme", "display": "Compression (Extreme)"} # Extreme, worst quality
]

def setup_directory(directory):
    """Create directory if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)
    return directory

def cosine_similarity(emb1, emb2):
    """Calculate cosine similarity between embeddings"""
    if emb1.ndim == 1:
        emb1 = emb1.reshape(1, -1)
    if emb2.ndim == 1:
        emb2 = emb2.reshape(1, -1)
    
    similarities = []
    for e1, e2 in zip(emb1, emb2):
        try:
            sim = 1 - cosine(e1, e2)
            similarities.append(sim)
        except:
            continue
    
    return np.mean(similarities) if similarities else 0.0

def calculate_audio_metrics(ref_path, deg_path):
    """Calculate PESQ and STOI scores"""
    try:
        ref, sr = librosa.load(ref_path, sr=16000)
        deg, _ = librosa.load(deg_path, sr=16000)
        
        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]
        
        pesq_score = pesq(sr, ref, deg, 'wb')
        stoi_score = stoi(ref, deg, sr, extended=False)
        
        return pesq_score, stoi_score
    except Exception as e:
        print(f"Error calculating metrics for {deg_path}: {str(e)}")
        return 0.0, 0.0

def process_audio_file(filename, audio_dir, openl3_dir, vggish_dir, results_dir):
    """Process a single audio file and calculate metrics for all its degraded versions"""
    try:
        # Create a results directory for this file
        file_results_dir = os.path.join(results_dir, filename)
        setup_directory(file_results_dir)
        
        # Path to clean audio
        clean_audio_path = os.path.join(audio_dir, f"{filename}.wav")
        if not os.path.exists(clean_audio_path):
            clean_audio_path = os.path.join(audio_dir, f"{filename}_clean.wav")
            if not os.path.exists(clean_audio_path):
                print(f"Clean audio not found for {filename}")
                return None
        
        # Load clean embeddings
        clean_openl3_path = os.path.join(openl3_dir, f"{filename}_clean.npy")
        clean_vggish_path = os.path.join(vggish_dir, f"{filename}_clean.npy")
        
        if not os.path.exists(clean_openl3_path) or not os.path.exists(clean_vggish_path):
            print(f"Clean embeddings not found for {filename}")
            return None
            
        clean_openl3 = np.load(clean_openl3_path)
        clean_vggish = np.load(clean_vggish_path)
        
        # Results for this file
        results = []
        
        # Process each degradation type
        for degradation in tqdm(DEGRADATIONS, desc=f"Processing {filename}", leave=False):
            suffix = degradation["suffix"]
            display_name = degradation["display"]
            
            # Paths to degraded files
            deg_audio_path = os.path.join(audio_dir, f"{filename}_{suffix}.wav")
            deg_openl3_path = os.path.join(openl3_dir, f"{filename}_{suffix}.npy")
            deg_vggish_path = os.path.join(vggish_dir, f"{filename}_{suffix}.npy")
            
            # Skip if files don't exist
            if not os.path.exists(deg_audio_path) or not os.path.exists(deg_openl3_path) or not os.path.exists(deg_vggish_path):
                print(f"Skipping {suffix} for {filename} - files not found")
                continue
                
            # Load degraded embeddings
            deg_openl3 = np.load(deg_openl3_path)
            deg_vggish = np.load(deg_vggish_path)
            
            # Calculate similarities
            openl3_sim = cosine_similarity(clean_openl3, deg_openl3)
            vggish_sim = cosine_similarity(clean_vggish, deg_vggish)
            
            # Calculate PESQ and STOI
            pesq_score, stoi_score = calculate_audio_metrics(clean_audio_path, deg_audio_path)
            
            # Store results
            results.append({
                'Filename': filename,
                'Degradation_Type': display_name,
                'Degradation_Suffix': suffix,
                'OpenL3_Similarity': openl3_sim,
                'VGGish_Similarity': vggish_sim,
                'PESQ_Score': pesq_score,
                'STOI_Score': stoi_score
            })
            
        # Create DataFrame and save results
        if results:
            results_df = pd.DataFrame(results)
            csv_path = os.path.join(file_results_dir, 'metrics.csv')
            results_df.to_csv(csv_path, index=False)
            print(f"Saved metrics for {filename} to {csv_path}")
            return results_df
        else:
            print(f"No results generated for {filename}")
            return None
            
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def batch_process(audio_dir, openl3_dir, vggish_dir, results_dir):
    """Process all audio files in batch"""
    # Get all unique filenames (without extensions and suffixes)
    filenames = set()
    
    # Look for audio files
    for file in os.listdir(audio_dir):
        if file.endswith('.wav'):
            # Remove extension
            name = os.path.splitext(file)[0]
            # Remove known suffixes
            for degradation in DEGRADATIONS:
                suffix = f"_{degradation['suffix']}"
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    break
            # Remove _clean suffix if present
            if name.endswith('_clean'):
                name = name[:-6]
            filenames.add(name)
    
    print(f"Found {len(filenames)} unique audio files to process")
    
    # Process each file
    all_results = []
    for filename in tqdm(filenames, desc="Processing files"):
        results_df = process_audio_file(filename, audio_dir, openl3_dir, vggish_dir, results_dir)
        if results_df is not None:
            all_results.append(results_df)
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_csv = os.path.join(results_dir, 'all_metrics.csv')
        combined_df.to_csv(combined_csv, index=False)
        print(f"\nSaved combined metrics to {combined_csv}")
        return combined_df
    else:
        print("No results were generated")
        return None

def main():
    parser = argparse.ArgumentParser(description="Calculate metrics for audio embeddings")
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR, help="Directory containing audio files")
    parser.add_argument("--openl3-dir", default=DEFAULT_OPENL3_DIR, help="Directory containing OpenL3 embeddings")
    parser.add_argument("--vggish-dir", default=DEFAULT_VGGISH_DIR, help="Directory containing VGGish embeddings")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="Directory to save results")
    
    args = parser.parse_args()
    
    # Setup directories
    audio_dir = args.audio_dir
    openl3_dir = args.openl3_dir
    vggish_dir = args.vggish_dir
    results_dir = setup_directory(args.results_dir)
    
    try:
        # Process all files
        batch_process(audio_dir, openl3_dir, vggish_dir, results_dir)
        print("\nMetrics calculation completed successfully!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
