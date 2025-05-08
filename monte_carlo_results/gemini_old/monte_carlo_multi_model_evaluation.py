import pandas as pd
import numpy as np
import random
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
import os
import json
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv
import sys

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now we can import from utils
from utils.chroma_utils import ChromaGO
from utils.openai_query import openai_chat
from utils.prompt_factory import make_user_prompt_with_score

from monte_carlo_rag_evaluation import MonteCarloRAGEvaluation

# Load environment variables
load_dotenv()

class GeminiMonteCarloEvaluation(MonteCarloRAGEvaluation):
    def __init__(self, config_path: str, output_dir: str = 'gemini_monte_carlo_results'):
        """Initialize the Gemini Monte Carlo evaluation."""
        super().__init__(config_path, output_dir)
        
        # Initialize Gemini API client
        genai.configure(api_key=os.getenv('GOOGLEAI_KEY'))
        
    def get_gemini_response(self, genes: List[str], context: str = None) -> Tuple[str, float]:
        """Get response from Gemini model."""
        if context:
            prompt = f"""Context from GO database:
{context}

Given the following set of genes: {', '.join(genes)}
Please analyze their biological function and provide:
1. The most prominent biological process these genes are involved in
2. A confidence score (0-1) for your assessment
Format your response as:
Process: [process name] | Score: [score]
[explanation]"""
        else:
            prompt = f"""Given the following set of genes: {', '.join(genes)}
            Please analyze their biological function and provide:
            1. The most prominent biological process these genes are involved in
            2. A confidence score (0-1) for your assessment
            Format your response as:
            Process: [process name] | Score: [score]
            [explanation]"""
        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            response_text = response.text
            score = self.extract_score(response_text)
            return response_text, score
        except Exception as e:
            print(f"Error with Gemini: {e}")
            return None, 0.0

    def run_gemini_analysis(self, genes: List[str]) -> Dict[str, float]:
        """Run both RAG and traditional analysis using Gemini model."""
        # Get relevant context from ChromaDB
        context = self.chroma_go.get_context_for_llm(genes)
        
        # Run traditional analysis
        _, traditional_score = self.get_gemini_response(genes)
        
        # Run RAG analysis
        _, rag_score = self.get_gemini_response(genes, context)
        
        return {
            'traditional_score': traditional_score,
            'rag_score': rag_score
        }

    def run_gemini_simulation(self, n_iterations: int = 100, gene_set_sizes: List[int] = [5, 10, 20]):
        """Run Monte Carlo simulation with Gemini model."""
        results = []
        
        for size in gene_set_sizes:
            print(f"\nRunning simulation for gene set size {size}")
            for i in tqdm(range(n_iterations)):
                # Generate random gene set
                genes, go_term = self.generate_random_gene_set(size)
                
                # Run analysis with Gemini
                start_time = time.time()
                scores = self.run_gemini_analysis(genes)
                end_time = time.time()
                
                # Store results
                results.append({
                    'iteration': i,
                    'gene_set_size': size,
                    'go_term': go_term,
                    'genes': ' '.join(genes),
                    'traditional_score': scores['traditional_score'],
                    'rag_score': scores['rag_score'],
                    'score_difference': scores['rag_score'] - scores['traditional_score'],
                    'processing_time': end_time - start_time
                })
                
                # Save intermediate results
                if (i + 1) % 10 == 0:
                    self.save_gemini_results(results)
                    
        return pd.DataFrame(results)

    def save_gemini_results(self, results: List[Dict]):
        """Save Gemini results to CSV file."""
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.output_dir, 'simulation_results_gemini.csv'), index=False)

    def analyze_gemini_results(self, results_df: pd.DataFrame):
        """Analyze and visualize Gemini simulation results."""
        # Create visualizations directory
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Score Distribution Comparison
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=results_df[['traditional_score', 'rag_score']])
        plt.title('Score Distribution: Traditional vs RAG (Gemini)')
        plt.ylabel('Score')
        plt.savefig(os.path.join(viz_dir, 'gemini_score_distribution.png'))
        plt.close()
        
        # 2. Score by Gene Set Size
        plt.figure(figsize=(15, 10))
        
        # Traditional scores
        plt.subplot(2, 2, 1)
        sns.boxplot(x='gene_set_size', y='traditional_score', data=results_df)
        plt.title('Traditional Scores by Gene Set Size')
        plt.xlabel('Gene Set Size')
        plt.ylabel('Score')
        
        # RAG scores
        plt.subplot(2, 2, 2)
        sns.boxplot(x='gene_set_size', y='rag_score', data=results_df)
        plt.title('RAG Scores by Gene Set Size')
        plt.xlabel('Gene Set Size')
        plt.ylabel('Score')
        
        # Score differences
        plt.subplot(2, 2, 3)
        sns.boxplot(x='gene_set_size', y='score_difference', data=results_df)
        plt.title('Score Differences by Gene Set Size')
        plt.xlabel('Gene Set Size')
        plt.ylabel('Score Difference (RAG - Traditional)')
        
        # Processing time
        plt.subplot(2, 2, 4)
        sns.boxplot(x='gene_set_size', y='processing_time', data=results_df)
        plt.title('Processing Time by Gene Set Size')
        plt.xlabel('Gene Set Size')
        plt.ylabel('Processing Time (seconds)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'gemini_analysis_by_size.png'))
        plt.close()
        
        # 3. Score Difference Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df['score_difference'], bins=20, kde=True)
        plt.title('Distribution of Score Differences (RAG - Traditional)')
        plt.xlabel('Score Difference')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(viz_dir, 'gemini_score_difference_distribution.png'))
        plt.close()
        
        # Statistical Analysis
        stats_results = {}
        for size in results_df['gene_set_size'].unique():
            size_data = results_df[results_df['gene_set_size'] == size]
            stats_results[size] = {
                'mean_traditional': size_data['traditional_score'].mean(),
                'mean_rag': size_data['rag_score'].mean(),
                'mean_difference': size_data['score_difference'].mean(),
                'std_traditional': size_data['traditional_score'].std(),
                'std_rag': size_data['rag_score'].std(),
                'std_difference': size_data['score_difference'].std(),
                'mean_processing_time': size_data['processing_time'].mean(),
                'std_processing_time': size_data['processing_time'].std()
            }
            
        # Save statistics
        stats_df = pd.DataFrame(stats_results).T
        stats_df.to_csv(os.path.join(self.output_dir, 'gemini_statistical_analysis.csv'))
        
        return stats_df

def main():
    # Initialize evaluation
    evaluator = GeminiMonteCarloEvaluation(
        config_path=os.path.join(parent_dir, 'jsonFiles/toyexample.json'),
        output_dir='monte_carlo_results'
    )
    
    # Run simulation
    results = evaluator.run_gemini_simulation(
        n_iterations=100,
        gene_set_sizes=[5, 10, 20]
    )
    
    # Analyze results
    stats = evaluator.analyze_gemini_results(results)
    print("\nGemini Statistical Analysis Results:")
    print(stats)

if __name__ == "__main__":
    main() 