import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

def analyze_results(results_file: str, output_dir: str):
    """Analyze and visualize simulation results."""
    # Read results
    df = pd.read_csv(results_file)
    
    # Create visualizations directory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Score Distribution by Method
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[['traditional_score', 'rag_score']])
    plt.title('Score Distribution by Method (DeepSeek)')
    plt.ylabel('Confidence Score')
    plt.savefig(os.path.join(viz_dir, 'score_distribution.png'))
    plt.close()
    
    # 2. Score Difference by Gene Set Size
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='gene_set_size', y='score_difference', data=df)
    plt.title('Score Difference by Gene Set Size (DeepSeek)')
    plt.xlabel('Gene Set Size')
    plt.ylabel('RAG Score - Traditional Score')
    plt.savefig(os.path.join(viz_dir, 'score_difference_by_size.png'))
    plt.close()
    
    # 3. Processing Time Analysis
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='gene_set_size', y='processing_time', data=df)
    plt.title('Processing Time by Gene Set Size (DeepSeek)')
    plt.xlabel('Gene Set Size')
    plt.ylabel('Processing Time (seconds)')
    plt.savefig(os.path.join(viz_dir, 'processing_time.png'))
    plt.close()
    
    # 4. Score Difference by GO Term
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='go_term', y='score_difference', data=df)
    plt.title('Score Difference by GO Term (DeepSeek)')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('GO Term')
    plt.ylabel('RAG Score - Traditional Score')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'score_difference_by_go_term.png'))
    plt.close()
    
    # 5. Scatter plot of Traditional vs RAG scores
    plt.figure(figsize=(10, 6))
    plt.scatter(df['traditional_score'], df['rag_score'], alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line
    plt.title('Traditional vs RAG Scores (DeepSeek)')
    plt.xlabel('Traditional Score')
    plt.ylabel('RAG Score')
    plt.savefig(os.path.join(viz_dir, 'score_comparison_scatter.png'))
    plt.close()
    
    # Statistical Analysis
    stats_results = {}
    for size in df['gene_set_size'].unique():
        size_data = df[df['gene_set_size'] == size]
        t_stat, p_value = stats.ttest_rel(size_data['rag_score'], size_data['traditional_score'])
        stats_results[size] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_difference': size_data['score_difference'].mean(),
            'std_difference': size_data['score_difference'].std(),
            'mean_traditional': size_data['traditional_score'].mean(),
            'mean_rag': size_data['rag_score'].mean(),
            'n_samples': len(size_data)
        }
    
    # Save statistics
    stats_df = pd.DataFrame(stats_results).T
    stats_df.to_csv(os.path.join(output_dir, 'statistical_analysis.csv'))
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nOverall Performance:")
    print(f"Mean Traditional Score: {df['traditional_score'].mean():.3f}")
    print(f"Mean RAG Score: {df['rag_score'].mean():.3f}")
    print(f"Mean Score Difference: {df['score_difference'].mean():.3f}")
    print(f"Standard Deviation of Score Difference: {df['score_difference'].std():.3f}")
    
    print("\nPerformance by Gene Set Size:")
    print(stats_df[['mean_traditional', 'mean_rag', 'mean_difference', 'p_value', 'n_samples']])
    
    return stats_df

if __name__ == "__main__":
    # Get the absolute path to the results file
    results_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'monte_carlo_results/deepseek_results/simulation_results.csv')
    
    # Get the output directory
    output_dir = os.path.dirname(results_file)
    
    # Run analysis
    stats = analyze_results(results_file, output_dir) 