import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(multi_model=False):
    """Load the simulation results data."""
    if multi_model:
        data_path = Path('../monte_carlo_results/simulation_results_multi_model.csv')
    else:
        data_path = Path('../monte_carlo_results/simulation_results.csv')
    return pd.read_csv(data_path)

def calculate_summary_statistics(df, multi_model=False):
    """Calculate summary statistics for scores and processing times."""
    if multi_model:
        stats = {
            'gene_set_size': [],
            'mean_traditional_score': [],
            'mean_rag_score': [],
            'mean_deepseek_score': [],
            'mean_gemini_score': [],
            'mean_processing_time': [],
            'std_processing_time': []
        }
        
        for size in df['gene_set_size'].unique():
            subset = df[df['gene_set_size'] == size]
            stats['gene_set_size'].append(size)
            stats['mean_traditional_score'].append(subset['traditional_score'].mean())
            stats['mean_rag_score'].append(subset['rag_score'].mean())
            stats['mean_deepseek_score'].append(subset['deepseek_score'].mean())
            stats['mean_gemini_score'].append(subset['gemini_score'].mean())
            stats['mean_processing_time'].append(subset['processing_time'].mean())
            stats['std_processing_time'].append(subset['processing_time'].std())
    else:
        stats = {
            'gene_set_size': [],
            'mean_traditional_score': [],
            'mean_rag_score': [],
            'mean_score_difference': [],
            'std_score_difference': [],
            'mean_processing_time': [],
            'std_processing_time': []
        }
        
        for size in df['gene_set_size'].unique():
            subset = df[df['gene_set_size'] == size]
            stats['gene_set_size'].append(size)
            stats['mean_traditional_score'].append(subset['traditional_score'].mean())
            stats['mean_rag_score'].append(subset['rag_score'].mean())
            stats['mean_score_difference'].append(subset['score_difference'].mean())
            stats['std_score_difference'].append(subset['score_difference'].std())
            stats['mean_processing_time'].append(subset['processing_time'].mean())
            stats['std_processing_time'].append(subset['processing_time'].std())
    
    return pd.DataFrame(stats)

def plot_score_comparison(df, multi_model=False):
    """Create plots comparing scores."""
    if multi_model:
        plt.figure(figsize=(15, 10))
        
        # Score distribution by model
        plt.subplot(2, 2, 1)
        sns.boxplot(data=df[['traditional_score', 'rag_score', 'deepseek_score', 'gemini_score']])
        plt.title('Score Distribution by Model')
        plt.xticks(rotation=45)
        
        # Score comparison by gene set size
        plt.subplot(2, 2, 2)
        sns.boxplot(x='gene_set_size', y='traditional_score', data=df)
        plt.title('Traditional Scores by Gene Set Size')
        
        plt.subplot(2, 2, 3)
        sns.boxplot(x='gene_set_size', y='rag_score', data=df)
        plt.title('RAG Scores by Gene Set Size')
        
        plt.subplot(2, 2, 4)
        sns.boxplot(x='gene_set_size', y='processing_time', data=df)
        plt.title('Processing Time by Gene Set Size')
    else:
        plt.figure(figsize=(12, 6))
        
        # Score comparison by gene set size
        plt.subplot(1, 2, 1)
        sns.boxplot(x='gene_set_size', y='score_difference', data=df)
        plt.title('Score Differences by Gene Set Size')
        plt.xlabel('Gene Set Size')
        plt.ylabel('Score Difference (RAG - Traditional)')
        
        # Processing time by gene set size
        plt.subplot(1, 2, 2)
        sns.boxplot(x='gene_set_size', y='processing_time', data=df)
        plt.title('Processing Time by Gene Set Size')
        plt.xlabel('Gene Set Size')
        plt.ylabel('Processing Time (seconds)')
    
    plt.tight_layout()
    plt.savefig('analysis/score_comparison.png')
    plt.close()

def plot_score_distributions(df, multi_model=False):
    """Create side-by-side plots of score distributions."""
    if multi_model:
        plt.figure(figsize=(15, 10))
        
        # Traditional scores
        plt.subplot(2, 2, 1)
        sns.histplot(df['traditional_score'], bins=20, kde=True)
        plt.title('Traditional Scores Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        
        # RAG scores
        plt.subplot(2, 2, 2)
        sns.histplot(df['rag_score'], bins=20, kde=True)
        plt.title('RAG Scores Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        
        # DeepSeek scores
        plt.subplot(2, 2, 3)
        sns.histplot(df['deepseek_score'], bins=20, kde=True)
        plt.title('DeepSeek Scores Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        
        # Gemini scores
        plt.subplot(2, 2, 4)
        sns.histplot(df['gemini_score'], bins=20, kde=True)
        plt.title('Gemini Scores Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
    else:
        plt.figure(figsize=(12, 6))
        
        # Traditional scores
        plt.subplot(1, 2, 1)
        sns.histplot(df['traditional_score'], bins=20, kde=True)
        plt.title('Traditional Scores Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        
        # RAG scores
        plt.subplot(1, 2, 2)
        sns.histplot(df['rag_score'], bins=20, kde=True)
        plt.title('RAG Scores Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('analysis/score_distributions.png')
    plt.close()

def analyze_score_distribution(df, multi_model=False):
    """Analyze the distribution of score differences."""
    if multi_model:
        plt.figure(figsize=(15, 10))
        
        # Compare all models
        plt.subplot(2, 2, 1)
        sns.boxplot(data=df[['traditional_score', 'rag_score', 'deepseek_score', 'gemini_score']])
        plt.title('Score Distribution Comparison')
        plt.xticks(rotation=45)
        
        # Compare RAG vs Traditional
        plt.subplot(2, 2, 2)
        sns.boxplot(data=df[['traditional_score', 'rag_score']])
        plt.title('RAG vs Traditional')
        
        # Compare DeepSeek vs Gemini
        plt.subplot(2, 2, 3)
        sns.boxplot(data=df[['deepseek_score', 'gemini_score']])
        plt.title('DeepSeek vs Gemini')
        
        # Compare all models by gene set size
        plt.subplot(2, 2, 4)
        df_melted = pd.melt(df, id_vars=['gene_set_size'], 
                           value_vars=['traditional_score', 'rag_score', 'deepseek_score', 'gemini_score'],
                           var_name='model', value_name='score')
        sns.boxplot(x='gene_set_size', y='score', hue='model', data=df_melted)
        plt.title('Score Distribution by Model and Gene Set Size')
        plt.xticks(rotation=45)
    else:
        plt.figure(figsize=(10, 6))
        
        # Histogram of score differences
        sns.histplot(df['score_difference'], bins=20, kde=True)
        plt.title('Distribution of Score Differences')
        plt.xlabel('Score Difference (RAG - Traditional)')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('analysis/score_distribution.png')
    plt.close()

def main():
    # Create analysis directory if it doesn't exist
    Path('analysis').mkdir(exist_ok=True)
    
    # Load data (both single and multi-model)
    df_single = load_data(multi_model=False)
    df_multi = load_data(multi_model=True)
    
    # Calculate summary statistics
    stats_single = calculate_summary_statistics(df_single, multi_model=False)
    stats_multi = calculate_summary_statistics(df_multi, multi_model=True)
    
    print("\nSingle Model Summary Statistics:")
    print(stats_single.to_string(index=False))
    
    print("\nMulti-Model Summary Statistics:")
    print(stats_multi.to_string(index=False))
    
    # Generate plots
    plot_score_comparison(df_single, multi_model=False)
    plot_score_comparison(df_multi, multi_model=True)
    plot_score_distributions(df_single, multi_model=False)
    plot_score_distributions(df_multi, multi_model=True)
    analyze_score_distribution(df_single, multi_model=False)
    analyze_score_distribution(df_multi, multi_model=True)
    
    # Save summary statistics
    stats_single.to_csv('analysis/summary_statistics_single.csv', index=False)
    stats_multi.to_csv('analysis/summary_statistics_multi.csv', index=False)
    
    print("\nAnalysis complete. Results saved in the 'analysis' directory.")

if __name__ == "__main__":
    main() 