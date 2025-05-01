import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from pesq import pesq
from pystoi import stoi
import pandas as pd

# Directory paths
AUDIO_DIR = "test_audio"
OPENL3_DIR = "openl3_embeddings"
VGGISH_DIR = "vggish_embeddings"
RESULTS_DIR = "results"

def setup_results_directory():
    """Create results directory"""
    os.makedirs(RESULTS_DIR, exist_ok=True)

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
        print(f"Error calculating metrics: {str(e)}")
        return 0.0, 0.0

def visualize_metrics(similarities, pesq_scores, stoi_scores, title_prefix):
    """Create and save visualization plots"""
    # Scatter plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Similarity vs PESQ
    ax1.scatter(similarities, pesq_scores, alpha=0.6)
    z1 = np.polyfit(similarities, pesq_scores, 1)
    p1 = np.poly1d(z1)
    ax1.plot(similarities, p1(similarities), "r--", alpha=0.8)
    ax1.set_xlabel("Embedding Similarity")
    ax1.set_ylabel("PESQ Score")
    ax1.set_title(f"{title_prefix} Similarity vs PESQ")
    ax1.grid(True)
    
    # Plot 2: Similarity vs STOI
    ax2.scatter(similarities, stoi_scores, alpha=0.6)
    z2 = np.polyfit(similarities, stoi_scores, 1)
    p2 = np.poly1d(z2)
    ax2.plot(similarities, p2(similarities), "r--", alpha=0.8)
    ax2.set_xlabel("Embedding Similarity")
    ax2.set_ylabel("STOI Score")
    ax2.set_title(f"{title_prefix} Similarity vs STOI")
    ax2.grid(True)
    
    plt.tight_layout()
    # Save scatter plots
    plt.savefig(os.path.join(RESULTS_DIR, f'{title_prefix}_scatter_plots.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Correlation matrix
    plt.figure(figsize=(8, 6))
    corr_matrix = np.corrcoef([similarities, pesq_scores, stoi_scores])
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt='.2f',
                xticklabels=['Similarity', 'PESQ', 'STOI'],
                yticklabels=['Similarity', 'PESQ', 'STOI'],
                cmap='coolwarm')
    plt.title(f"{title_prefix} Correlation Matrix")
    
    # Save correlation matrix
    plt.savefig(os.path.join(RESULTS_DIR, f'{title_prefix}_correlation_matrix.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return corr_matrix

def visualize_noise_effects(noise_levels, similarities_openl3, similarities_vggish, 
                          pesq_scores, stoi_scores):
    """Create plots showing how noise levels affect different metrics"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Noise Level vs Similarities
    plt.subplot(2, 1, 1)
    plt.plot(noise_levels, similarities_openl3, 'o-', label='OpenL3 Similarity', color='blue')
    plt.plot(noise_levels, similarities_vggish, 'o-', label='VGGish Similarity', color='red')
    plt.xlabel('Noise Level')
    plt.ylabel('Similarity Score')
    plt.title('Noise Level vs Embedding Similarities')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    
    # Plot 2: Noise Level vs PESQ and STOI
    plt.subplot(2, 1, 2)
    plt.plot(noise_levels, pesq_scores, 'o-', label='PESQ', color='green')
    plt.plot(noise_levels, stoi_scores, 'o-', label='STOI', color='purple')
    plt.xlabel('Noise Level')
    plt.ylabel('Score')
    plt.title('Noise Level vs Perceptual Metrics')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    
    plt.tight_layout()
    # Save the plot
    plt.savefig(os.path.join(RESULTS_DIR, 'noise_level_effects.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_tables(results_df):
    """Create and save comparison tables as both CSV and visual formats"""
    # Create figure for bar charts
    plt.figure(figsize=(15, 10))
    
    # Plot 1: PESQ and STOI vs Noise Level
    plt.subplot(2, 1, 1)
    bar_width = 0.35
    index = np.arange(len(results_df['Noise_Level']))
    
    plt.bar(index, results_df['PESQ_Score'], bar_width, 
            label='PESQ', color='green', alpha=0.7)
    plt.bar(index + bar_width, results_df['STOI_Score'], bar_width,
            label='STOI', color='purple', alpha=0.7)
    
    plt.xlabel('Noise Level')
    plt.ylabel('Score')
    plt.title('PESQ and STOI Scores vs Noise Level')
    plt.xticks(index + bar_width/2, [f'{x:.3f}' for x in results_df['Noise_Level']])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Similarities vs Noise Level
    plt.subplot(2, 1, 2)
    plt.bar(index, results_df['OpenL3_Similarity'], bar_width,
            label='OpenL3', color='blue', alpha=0.7)
    plt.bar(index + bar_width, results_df['VGGish_Similarity'], bar_width,
            label='VGGish', color='red', alpha=0.7)
    
    plt.xlabel('Noise Level')
    plt.ylabel('Similarity Score')
    plt.title('Embedding Similarities vs Noise Level')
    plt.xticks(index + bar_width/2, [f'{x:.3f}' for x in results_df['Noise_Level']])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'comparison_bars.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create comparison tables
    # Table 1: PESQ and STOI
    perceptual_table = pd.DataFrame({
        'Noise_Level': results_df['Noise_Level'],
        'PESQ': results_df['PESQ_Score'],
        'STOI': results_df['STOI_Score']
    })
    
    # Table 2: Similarities
    similarity_table = pd.DataFrame({
        'Noise_Level': results_df['Noise_Level'],
        'OpenL3': results_df['OpenL3_Similarity'],
        'VGGish': results_df['VGGish_Similarity']
    })
    
    # Save tables as CSV
    perceptual_table.to_csv(os.path.join(RESULTS_DIR, 'perceptual_metrics_table.csv'), 
                           index=False, float_format='%.4f')
    similarity_table.to_csv(os.path.join(RESULTS_DIR, 'similarity_metrics_table.csv'), 
                           index=False, float_format='%.4f')
    
    # Add tables to the report
    with open(os.path.join(RESULTS_DIR, 'analysis_report.txt'), 'a') as f:
        f.write("\n\nPerceptual Metrics Table:\n")
        f.write("-" * 40 + "\n")
        f.write(perceptual_table.to_string(index=False, float_format=lambda x: '%.4f' % x))
        
        f.write("\n\nSimilarity Metrics Table:\n")
        f.write("-" * 40 + "\n")
        f.write(similarity_table.to_string(index=False, float_format=lambda x: '%.4f' % x))

def save_analysis_report(openl3_corr, vggish_corr, results_df):
    """Save analysis report as text file"""
    report_path = os.path.join(RESULTS_DIR, 'analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("Audio Quality Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("OpenL3 Correlations:\n")
        f.write(f"Similarity vs PESQ: {openl3_corr[0,1]:.3f}\n")
        f.write(f"Similarity vs STOI: {openl3_corr[0,2]:.3f}\n\n")
        
        f.write("VGGish Correlations:\n")
        f.write(f"Similarity vs PESQ: {vggish_corr[0,1]:.3f}\n")
        f.write(f"Similarity vs STOI: {vggish_corr[0,2]:.3f}\n\n")
        
        f.write("Statistical Summary:\n")
        f.write(results_df.describe().to_string())

def main():
    try:
        # Create results directory
        setup_results_directory()
        print(f"\nResults will be saved in: {RESULTS_DIR}")
        
        # Lists to store results
        similarities_openl3 = []
        similarities_vggish = []
        pesq_scores = []
        stoi_scores = []
        noise_levels = []
        
        # Load clean audio embeddings
        clean_openl3 = np.load(os.path.join(OPENL3_DIR, "test_audio_clean.npy"))
        clean_vggish = np.load(os.path.join(VGGISH_DIR, "test_audio_clean.npy"))
        
        # Process each degraded version
        for i in range(1, 6):
            try:
                # Load degraded embeddings
                deg_openl3 = np.load(os.path.join(OPENL3_DIR, f"test_audio_degraded_{i}.npy"))
                deg_vggish = np.load(os.path.join(VGGISH_DIR, f"test_audio_degraded_{i}.npy"))
                
                # Calculate similarities
                sim_openl3 = cosine_similarity(clean_openl3, deg_openl3)
                sim_vggish = cosine_similarity(clean_vggish, deg_vggish)
                
                # Calculate PESQ and STOI
                clean_path = os.path.join(AUDIO_DIR, "test_audio.wav")
                deg_path = os.path.join(AUDIO_DIR, f"test_audio_degraded_{i}.wav")
                pesq_score, stoi_score = calculate_audio_metrics(clean_path, deg_path)
                
                # Store results
                similarities_openl3.append(sim_openl3)
                similarities_vggish.append(sim_vggish)
                pesq_scores.append(pesq_score)
                stoi_scores.append(stoi_score)
                noise_levels.append([0.001, 0.002, 0.005, 0.01, 0.02][i-1])
                
                print(f"\nProcessed degraded version {i}:")
                print(f"OpenL3 Similarity: {sim_openl3:.3f}")
                print(f"VGGish Similarity: {sim_vggish:.3f}")
                print(f"PESQ Score: {pesq_score:.3f}")
                print(f"STOI Score: {stoi_score:.3f}")
                
            except Exception as e:
                print(f"Error processing degraded version {i}: {str(e)}")
                continue
        
        # Create DataFrame
        results_df = pd.DataFrame({
            'Noise_Level': noise_levels,
            'OpenL3_Similarity': similarities_openl3,
            'VGGish_Similarity': similarities_vggish,
            'PESQ_Score': pesq_scores,
            'STOI_Score': stoi_scores
        })
        
        # Save results to CSV
        csv_path = os.path.join(RESULTS_DIR, 'evaluation_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Generate and save visualizations
        print("\nGenerating visualizations...")
        openl3_corr = visualize_metrics(similarities_openl3, pesq_scores, stoi_scores, "OpenL3")
        vggish_corr = visualize_metrics(similarities_vggish, pesq_scores, stoi_scores, "VGGish")
        
        # Generate and save noise level effects plot
        print("Generating noise level effects plot...")
        visualize_noise_effects(noise_levels, similarities_openl3, similarities_vggish, 
                              pesq_scores, stoi_scores)
        
        # Generate comparison tables and bar charts
        print("\nGenerating comparison tables and bar charts...")
        create_comparison_tables(results_df)
        
        # Save analysis report
        save_analysis_report(openl3_corr, vggish_corr, results_df)
        
        # Add noise level analysis to the report
        with open(os.path.join(RESULTS_DIR, 'analysis_report.txt'), 'a') as f:
            f.write("\n\nNoise Level Analysis:\n")
            f.write("-" * 20 + "\n")
            f.write("Impact of noise on metrics:\n")
            for i, noise in enumerate(noise_levels):
                f.write(f"\nNoise Level {noise:.3f}:\n")
                f.write(f"OpenL3 Similarity: {similarities_openl3[i]:.3f}\n")
                f.write(f"VGGish Similarity: {similarities_vggish[i]:.3f}\n")
                f.write(f"PESQ Score: {pesq_scores[i]:.3f}\n")
                f.write(f"STOI Score: {stoi_scores[i]:.3f}\n")
        
        print(f"\nAnalysis report saved in: {RESULTS_DIR}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main()
