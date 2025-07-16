import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from tqdm import tqdm

# Default directory paths
DEFAULT_RESULTS_DIR = "results"

def setup_directory(directory):
    """Create directory if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)
    return directory

def visualize_metrics(results_df, output_dir, prefix=""):
    """Create and save visualization plots for embedding vs perceptual metrics"""
    # Create separate plots for OpenL3 and VGGish
    for emb_type in ['OpenL3', 'VGGish']:
        similarities = results_df[f'{emb_type}_Similarity'].values
        pesq_scores = results_df['PESQ_Score'].values
        stoi_scores = results_df['STOI_Score'].values
        
        # Scatter plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Similarity vs PESQ
        ax1.scatter(similarities, pesq_scores, alpha=0.6)
        if len(similarities) > 1:  # Need at least 2 points for polyfit
            z1 = np.polyfit(similarities, pesq_scores, 1)
            p1 = np.poly1d(z1)
            ax1.plot(similarities, p1(similarities), "r--", alpha=0.8)
        ax1.set_xlabel(f"{emb_type} Similarity")
        ax1.set_ylabel("PESQ Score")
        ax1.set_title(f"{emb_type} Similarity vs PESQ")
        ax1.grid(True)
        
        # Plot 2: Similarity vs STOI
        ax2.scatter(similarities, stoi_scores, alpha=0.6)
        if len(similarities) > 1:  # Need at least 2 points for polyfit
            z2 = np.polyfit(similarities, stoi_scores, 1)
            p2 = np.poly1d(z2)
            ax2.plot(similarities, p2(similarities), "r--", alpha=0.8)
        ax2.set_xlabel(f"{emb_type} Similarity")
        ax2.set_ylabel("STOI Score")
        ax2.set_title(f"{emb_type} Similarity vs STOI")
        ax2.grid(True)
        
        plt.tight_layout()
        # Save scatter plots
        plot_filename = f'{prefix}{emb_type}_scatter_plots.png'
        plt.savefig(os.path.join(output_dir, plot_filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Correlation matrix
        if len(similarities) > 1:  # Need at least 2 points for correlation
            plt.figure(figsize=(8, 6))
            corr_matrix = np.corrcoef([similarities, pesq_scores, stoi_scores])
            sns.heatmap(corr_matrix, 
                        annot=True, 
                        fmt='.2f',
                        xticklabels=['Similarity', 'PESQ', 'STOI'],
                        yticklabels=['Similarity', 'PESQ', 'STOI'],
                        cmap='coolwarm')
            plt.title(f"{emb_type} Correlation Matrix")
            
            # Save correlation matrix
            corr_filename = f'{prefix}{emb_type}_correlation_matrix.png'
            plt.savefig(os.path.join(output_dir, corr_filename), dpi=300, bbox_inches='tight')
            plt.close()

def visualize_degradation_comparison(results_df, output_dir, prefix=""):
    """Create plots comparing different degradation types"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Degradation Type vs Similarities
    plt.subplot(2, 1, 1)
    
    # Set up bar chart
    x = np.arange(len(results_df))
    width = 0.35
    
    plt.bar(x - width/2, results_df['OpenL3_Similarity'], width, label='OpenL3', color='blue', alpha=0.7)
    plt.bar(x + width/2, results_df['VGGish_Similarity'], width, label='VGGish', color='red', alpha=0.7)
    
    plt.xlabel('Degradation Type')
    plt.ylabel('Similarity Score')
    plt.title('Embedding Similarities by Degradation Type')
    plt.xticks(x, results_df['Degradation_Type'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Degradation Type vs PESQ and STOI
    plt.subplot(2, 1, 2)
    
    plt.bar(x - width/2, results_df['PESQ_Score'], width, label='PESQ', color='green', alpha=0.7)
    plt.bar(x + width/2, results_df['STOI_Score'], width, label='STOI', color='purple', alpha=0.7)
    
    plt.xlabel('Degradation Type')
    plt.ylabel('Score')
    plt.title('Perceptual Metrics by Degradation Type')
    plt.xticks(x, results_df['Degradation_Type'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Move legend outside
    axes[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for legends
    plt.savefig(os.path.join(output_dir, f'{prefix}degradation_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def visualize_quality_comparison(results_df, output_dir, prefix=""):
    """Create plots comparing different quality levels for each degradation type"""
    # Extract unique degradation types and quality levels
    degradation_types = results_df['Degradation_Type'].str.split(' ', n=1).str[0].unique()
    
    # Create a figure with subplots for each metric
    metrics = ['OpenL3_Similarity', 'VGGish_Similarity', 'PESQ_Score', 'STOI_Score']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Consistent colors for quality levels
    colors = {'Low': 'green', 'High': 'orange', 'Extreme': 'red'}
    quality_order = ['Low', 'High', 'Extreme']  # Always plot in this order
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        x = np.arange(len(degradation_types))
        width = 0.25
        
        for j, quality in enumerate(quality_order):
            values = []
            for dtype in degradation_types:
                # Find row for this degradation type and quality
                row = results_df[results_df['Degradation_Type'] == f"{dtype} ({quality})"]
                val = row[metric].values[0] if not row.empty else 0
                values.append(val)
            position = x + (j-1)*width
            ax.bar(position, values, width, label=f'{quality} Quality',
                   color=colors[quality], alpha=0.7)
        
        ax.set_xlabel('Degradation Type')
        ax.set_ylabel(metric.replace('_', ' '))
        ax.set_title(f'{metric.replace("_", " ")} by Quality Level')
        ax.set_xticks(x)
        ax.set_xticklabels(degradation_types)
        # Move legend outside
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for legends
    plt.savefig(os.path.join(output_dir, f'{prefix}quality_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_tables(results_df, output_dir, prefix=""):
    """Create and save comparison tables as CSV files"""
    # Table 1: PESQ and STOI
    perceptual_table = pd.DataFrame({
        'Degradation_Type': results_df['Degradation_Type'],
        'PESQ': results_df['PESQ_Score'],
        'STOI': results_df['STOI_Score']
    })
    
    # Table 2: Similarities
    similarity_table = pd.DataFrame({
        'Degradation_Type': results_df['Degradation_Type'],
        'OpenL3': results_df['OpenL3_Similarity'],
        'VGGish': results_df['VGGish_Similarity']
    })
    
    # Save tables as CSV
    perceptual_table.to_csv(os.path.join(output_dir, f'{prefix}perceptual_metrics_table.csv'), 
                           index=False, float_format='%.4f')
    similarity_table.to_csv(os.path.join(output_dir, f'{prefix}similarity_metrics_table.csv'), 
                           index=False, float_format='%.4f')
    
    # Create analysis report
    report_path = os.path.join(output_dir, f'{prefix}analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write("Audio Quality Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Calculate correlations
        openl3_pesq_corr = np.corrcoef(results_df['OpenL3_Similarity'], results_df['PESQ_Score'])[0,1]
        openl3_stoi_corr = np.corrcoef(results_df['OpenL3_Similarity'], results_df['STOI_Score'])[0,1]
        vggish_pesq_corr = np.corrcoef(results_df['VGGish_Similarity'], results_df['PESQ_Score'])[0,1]
        vggish_stoi_corr = np.corrcoef(results_df['VGGish_Similarity'], results_df['STOI_Score'])[0,1]
        
        f.write("Correlations:\n")
        f.write(f"OpenL3 vs PESQ: {openl3_pesq_corr:.3f}\n")
        f.write(f"OpenL3 vs STOI: {openl3_stoi_corr:.3f}\n")
        f.write(f"VGGish vs PESQ: {vggish_pesq_corr:.3f}\n")
        f.write(f"VGGish vs STOI: {vggish_stoi_corr:.3f}\n\n")
        
        f.write("Statistical Summary:\n")
        f.write(results_df.describe().to_string())
        
        f.write("\n\nPerceptual Metrics Table:\n")
        f.write("-" * 40 + "\n")
        f.write(perceptual_table.to_string(index=False, float_format=lambda x: '%.4f' % x))
        
        f.write("\n\nSimilarity Metrics Table:\n")
        f.write("-" * 40 + "\n")
        f.write(similarity_table.to_string(index=False, float_format=lambda x: '%.4f' % x))
        
        # Add detailed analysis by degradation type
        f.write("\n\nDetailed Analysis by Degradation Type:\n")
        f.write("-" * 40 + "\n")
        
        for deg_type in results_df['Degradation_Type'].unique():
            subset = results_df[results_df['Degradation_Type'] == deg_type]
            if not subset.empty:
                row = subset.iloc[0]
                f.write(f"\n{deg_type}:\n")
                f.write(f"OpenL3 Similarity: {row['OpenL3_Similarity']:.3f}\n")
                f.write(f"VGGish Similarity: {row['VGGish_Similarity']:.3f}\n")
                f.write(f"PESQ Score: {row['PESQ_Score']:.3f}\n")
                f.write(f"STOI Score: {row['STOI_Score']:.3f}\n")

def process_file_results(file_metrics_path, output_dir):
    """Process metrics for a single file and generate visualizations"""
    try:
        # Load metrics
        results_df = pd.read_csv(file_metrics_path)
        
        # Get filename from path
        filename = os.path.basename(os.path.dirname(file_metrics_path))
        
        # Create output directory
        file_output_dir = os.path.join(output_dir, filename)
        setup_directory(file_output_dir)
        
        # Generate visualizations with filename prefix
        prefix = f"{filename}_"
        
        visualize_metrics(results_df, file_output_dir, prefix)
        visualize_degradation_comparison(results_df, file_output_dir, prefix)
        visualize_quality_comparison(results_df, file_output_dir, prefix)
        create_comparison_tables(results_df, file_output_dir, prefix)
        
        return True
    except Exception as e:
        print(f"Error processing {file_metrics_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_all_results(results_dir, output_dir=None):
    """Process all metrics files and generate visualizations"""
    if output_dir is None:
        output_dir = os.path.join(results_dir, "visualizations")
    
    setup_directory(output_dir)
    
    # Look for all_metrics.csv first
    all_metrics_path = os.path.join(results_dir, "all_metrics.csv")
    if os.path.exists(all_metrics_path):
        print(f"Processing combined metrics from {all_metrics_path}")
        results_df = pd.read_csv(all_metrics_path)
        
        # Generate overall visualizations
        visualize_metrics(results_df, output_dir)
        visualize_degradation_comparison(results_df, output_dir)
        visualize_quality_comparison(results_df, output_dir)
        create_comparison_tables(results_df, output_dir)
        
        # Process by file
        for filename in results_df['Filename'].unique():
            file_df = results_df[results_df['Filename'] == filename]
            file_output_dir = os.path.join(output_dir, filename)
            setup_directory(file_output_dir)
            
            prefix = f"{filename}_"
            visualize_metrics(file_df, file_output_dir, prefix)
            visualize_degradation_comparison(file_df, file_output_dir, prefix)
            visualize_quality_comparison(file_df, file_output_dir, prefix)
            create_comparison_tables(file_df, file_output_dir, prefix)
    else:
        # Look for individual metrics files
        processed = 0
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if file == "metrics.csv":
                    metrics_path = os.path.join(root, file)
                    print(f"Processing {metrics_path}")
                    if process_file_results(metrics_path, output_dir):
                        processed += 1
        
        print(f"Processed {processed} metrics files")

def main():
    parser = argparse.ArgumentParser(description="Visualize audio embedding metrics")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="Directory containing metrics results")
    parser.add_argument("--output-dir", help="Directory to save visualizations (default: results_dir/visualizations)")
    
    args = parser.parse_args()
    
    results_dir = args.results_dir
    output_dir = args.output_dir
    
    try:
        process_all_results(results_dir, output_dir)
        print("\nVisualization completed successfully!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
