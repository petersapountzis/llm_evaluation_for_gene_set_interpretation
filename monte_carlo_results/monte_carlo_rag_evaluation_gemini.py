import os
import sys
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import random
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
from utils.chroma_utils import ChromaGO
from utils.genai_query import query_genai_model
from utils.prompt_factory import make_user_prompt_with_score
import json
from tqdm import tqdm

class MonteCarloRAGEvaluationGemini:
    def __init__(self, config_path: str, output_dir: str = 'monte_carlo_results/gemini_results'):
        """Initialize the Monte Carlo evaluation."""
        # Convert config path to absolute path if it's relative
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config_path)
        self.config_path = config_path
        self.output_dir = output_dir
        self.chroma_go = ChromaGO()
        self.load_config()
        self.setup_output_directory()
        self.load_go_terms()
        
    def load_config(self):
        """Load configuration from JSON file."""
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
            
    def setup_output_directory(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_go_terms(self):
        """Load GO terms and their associated genes from the database."""
        # Load GO terms from the database
        go_terms_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/go_terms.csv')
        self.go_terms = pd.read_csv(go_terms_path)
        
        # Create a dictionary mapping GO terms to their genes
        self.go_term_genes = {}
        for _, row in self.go_terms.iterrows():
            go_term = row['GO']
            genes = row['Genes'].split()
            self.go_term_genes[go_term] = genes
            
        # Get list of all unique GO terms
        self.available_go_terms = list(self.go_term_genes.keys())
        
    def generate_random_gene_set(self, size: int, process: str = None) -> Tuple[List[str], str]:
        """Generate a random gene set of specified size from a random GO term."""
        # Select a random GO term
        if process is None:
            selected_go = random.choice(self.available_go_terms)
        else:
            # If process is specified, find a GO term that contains that process name
            matching_terms = [go for go in self.available_go_terms if process.lower() in go.lower()]
            if not matching_terms:
                raise ValueError(f"No GO terms found containing process: {process}")
            selected_go = random.choice(matching_terms)
            
        # Get genes associated with the selected GO term
        available_genes = self.go_term_genes[selected_go]
        
        # If we need more genes than available, use all available genes
        if size >= len(available_genes):
            return available_genes, selected_go
            
        # Otherwise, randomly sample the requested number of genes
        selected_genes = random.sample(available_genes, size)
        return selected_genes, selected_go
        
    def run_analysis(self, genes: List[str]) -> Tuple[float, float]:
        """Run both RAG and traditional analysis on a gene set."""
        # Get relevant context from ChromaDB
        context = self.chroma_go.get_context_for_llm(genes)
        
        # Create prompts
        traditional_prompt = make_user_prompt_with_score(genes=genes)
        rag_prompt = f"""Context from GO database:
{context}

{traditional_prompt}"""
        
        # Ensure log file path is absolute
        log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), self.config['LOG_NAME'])
        
        # Run traditional analysis
        traditional_response, _ = query_genai_model(
            prompt=traditional_prompt,
            model='gemini-2.0-flash-lite',
            temperature=self.config['TEMP'],
            max_tokens=self.config['MAX_TOKENS'],
            LOG_FILE=log_file
        )
        
        # Run RAG analysis
        rag_response, _ = query_genai_model(
            prompt=rag_prompt,
            model='gemini-2.0-flash-lite',
            temperature=self.config['TEMP'],
            max_tokens=self.config['MAX_TOKENS'],
            LOG_FILE=log_file
        )
        
        # Extract scores
        traditional_score = self.extract_score(traditional_response)
        rag_score = self.extract_score(rag_response)
        
        return traditional_score, rag_score
        
    def extract_score(self, response: str) -> float:
        """Extract confidence score from response."""
        try:
            # First try to find score in the new format "Process: [name] | Score: [score]"
            if "Process:" in response and "| Score:" in response:
                first_line = response.split('\n')[0]
                if "| Score:" in first_line:
                    score_str = first_line.split("| Score:")[1].strip()
                    return float(score_str)
            
            # Fallback to old format if new format not found
            if "Process:" in response:
                process_line = response.split('\n')[0]
                if '(' in process_line and ')' in process_line:
                    score_str = process_line.split('(')[-1].split(')')[0]
                    return float(score_str)
            
            return 0.0
        except:
            return 0.0
            
    def run_simulation(self, n_iterations: int = 100, gene_set_sizes: List[int] = [5, 10, 20]):
        """Run Monte Carlo simulation with specified parameters."""
        results = []
        
        for size in gene_set_sizes:
            print(f"\nRunning simulation for gene set size {size}")
            for i in tqdm(range(n_iterations)):
                # Generate random gene set
                genes, go_term = self.generate_random_gene_set(size)
                
                # Run analysis
                start_time = time.time()
                traditional_score, rag_score = self.run_analysis(genes)
                end_time = time.time()
                
                # Store results
                results.append({
                    'iteration': i,
                    'gene_set_size': size,
                    'go_term': go_term,
                    'genes': ' '.join(genes),
                    'traditional_score': traditional_score,
                    'rag_score': rag_score,
                    'score_difference': rag_score - traditional_score,
                    'processing_time': end_time - start_time
                })
                
                # Save intermediate results
                if (i + 1) % 10 == 0:  # Save every 10 iterations
                    self.save_results(results)
                    
        return pd.DataFrame(results)
        
    def save_results(self, results: List[Dict]):
        """Save results to CSV file."""
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.output_dir, 'gemini_simulation_results.csv'), index=False)
        
    def analyze_results(self, results_df: pd.DataFrame):
        """Analyze and visualize simulation results."""
        # Create visualizations directory
        viz_dir = os.path.join(self.output_dir, 'visualizations_gemini')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Score Distribution by Method
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=results_df[['traditional_score', 'rag_score']])
        plt.title('Score Distribution by Method (Gemini)')
        plt.savefig(os.path.join(viz_dir, 'score_distribution.png'))
        plt.close()
        
        # 2. Score Difference by Gene Set Size
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='gene_set_size', y='score_difference', data=results_df)
        plt.title('Score Difference by Gene Set Size (Gemini)')
        plt.savefig(os.path.join(viz_dir, 'score_difference_by_size.png'))
        plt.close()
        
        # 3. Processing Time Analysis
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='gene_set_size', y='processing_time', data=results_df)
        plt.title('Processing Time by Gene Set Size (Gemini)')
        plt.savefig(os.path.join(viz_dir, 'processing_time.png'))
        plt.close()
        
        # 4. Score Difference by GO Term
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='go_term', y='score_difference', data=results_df)
        plt.title('Score Difference by GO Term (Gemini)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'score_difference_by_go_term.png'))
        plt.close()
        
        # Statistical Analysis
        stats_results = {}
        for size in results_df['gene_set_size'].unique():
            size_data = results_df[results_df['gene_set_size'] == size]
            t_stat, p_value = stats.ttest_rel(size_data['rag_score'], size_data['traditional_score'])
            stats_results[size] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'mean_difference': size_data['score_difference'].mean(),
                'std_difference': size_data['score_difference'].std()
            }
            
        # Save statistics
        stats_df = pd.DataFrame(stats_results).T
        stats_df.to_csv(os.path.join(self.output_dir, 'statistical_analysis_gemini.csv'))
        
        return stats_df

def main():
    # Initialize evaluation
    evaluator = MonteCarloRAGEvaluationGemini(
        config_path='jsonFiles/toyexample.json',
        output_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'monte_carlo_results/gemini_results')
    )
    
    # Run simulation
    results = evaluator.run_simulation(
        n_iterations=100,  # Number of iterations per gene set size
        gene_set_sizes=[5, 10, 20]  # Different gene set sizes to test
    )
    
    # Analyze results
    stats = evaluator.analyze_results(results)
    print("\nStatistical Analysis Results:")
    print(stats)

if __name__ == "__main__":
    main() 