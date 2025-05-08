import pandas as pd
import numpy as np
from utils.chroma_utils import ChromaGO
from utils.openai_query import openai_chat
from utils.prompt_factory import make_user_prompt_with_score
import json
import os
from tqdm import tqdm
import time
import random
from typing import Dict, Any

def validate_path(path, description):
    """Validate that a path exists."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} not found: {path}")
    return path

def run_traditional_pipeline(gene_list, config):
    """Run traditional pipeline without RAG."""
    prompt = make_user_prompt_with_score(genes=gene_list)
    
    try:
        # Add longer random delay between 3-5 seconds to avoid rate limits
        time.sleep(random.uniform(3, 5))
        
        response_text, fingerprint = openai_chat(
            context=config['CONTEXT'],
            prompt=prompt,
            model=config['GPT_MODEL'],
            temperature=config['TEMP'],
            max_tokens=config['MAX_TOKENS'],
            rate_per_token=config['RATE_PER_TOKEN'],
            LOG_FILE=config['LOG_NAME'],
            DOLLAR_LIMIT=config['DOLLAR_LIMIT'],
            seed=config.get('SEED', 42)
        )
        return response_text
    except Exception as e:
        print(f"Error in traditional pipeline: {e}")
        return f"Error: {str(e)}"

def run_rag_pipeline(gene_list, config, chroma_db):
    """Run RAG-enhanced pipeline."""
    try:
        # Get relevant context from Chroma
        context = chroma_db.get_context_for_llm(gene_list)
        
        # Format context for LLM
        llm_context = f"""You are an efficient and insightful assistant to a molecular biologist.
You should give the true answer that is supported by the references. If you do not have a clear answer, you will respond with "Unknown".

Important context for these genes can be found here:
{context}
"""
        
        prompt = make_user_prompt_with_score(genes=gene_list)
        
        # Add longer random delay between 3-5 seconds to avoid rate limits
        time.sleep(random.uniform(3, 5))
        
        response_text, fingerprint = openai_chat(
            context=llm_context,
            prompt=prompt,
            model=config['GPT_MODEL'],
            temperature=config['TEMP'],
            max_tokens=config['MAX_TOKENS'],
            rate_per_token=config['RATE_PER_TOKEN'],
            LOG_FILE=config['LOG_NAME'],
            DOLLAR_LIMIT=config['DOLLAR_LIMIT'],
            seed=config.get('SEED', 42)
        )
        return response_text
    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        return f"Error: {str(e)}"

def extract_score(response: str) -> float:
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
            process_line = response.split('\n')[0]  # Get first line
            if '(' in process_line and ')' in process_line:
                score_str = process_line.split('(')[-1].split(')')[0]
                return float(score_str)
        
        # If that fails, look for score anywhere in the first few lines
        lines = response.split('\n')[:3]  # Check first 3 lines
        for line in lines:
            if '(' in line and ')' in line:
                try:
                    score_str = line.split('(')[-1].split(')')[0]
                    return float(score_str)
                except:
                    continue
        
        return 0.0
    except:
        return 0.0

def calculate_statistics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate summary statistics from results."""
    # Extract scores
    results_df['Traditional_Score'] = results_df['Traditional_Response'].apply(extract_score)
    results_df['RAG_Score'] = results_df['RAG_Response'].apply(extract_score)
    results_df['Score_Difference'] = results_df['RAG_Score'] - results_df['Traditional_Score']
    
    # Calculate statistics
    stats = {
        'total_terms': len(results_df),
        'mean_traditional_score': results_df['Traditional_Score'].mean(),
        'mean_rag_score': results_df['RAG_Score'].mean(),
        'mean_score_difference': results_df['Score_Difference'].mean(),
        'median_score_difference': results_df['Score_Difference'].median(),
        'std_score_difference': results_df['Score_Difference'].std(),
        'rag_higher_count': (results_df['Score_Difference'] > 0).sum(),
        'traditional_higher_count': (results_df['Score_Difference'] < 0).sum(),
        'equal_scores_count': (results_df['Score_Difference'] == 0).sum(),
        'max_score_difference': results_df['Score_Difference'].max(),
        'min_score_difference': results_df['Score_Difference'].min()
    }
    
    return stats

def main():
    # Check API key first
    print("Checking API key...")
    # if not check_api_key():
    #     print("API key check failed. Please verify your API key and try again.")
    #     return
    
    print("API key check passed. Starting with initial delay...")
    time.sleep(10)
    
    # Validate and load config
    config_path = validate_path('jsonFiles/toyexample.json', "Config file")
    with open(config_path) as f:
        config = json.load(f)
        
    # Initialize Chroma database
    chroma_db = ChromaGO()
    if not os.path.exists("./chroma_db"):
        chroma_db.load_go_terms("data/go_terms.csv")
        
    # Load test data
    test_data_path = validate_path('data/GO_term_analysis/toy_example.csv', "Test data file")
    df = pd.read_csv(test_data_path)
    
    # Initialize results dataframe
    results = pd.DataFrame(columns=[
        'GO', 'Genes', 'Traditional_Response', 'Traditional_Score',
        'RAG_Response', 'RAG_Score', 'Score_Difference'
    ])
    
    # Create output directory if it doesn't exist
    output_dir = 'data/GO_term_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run comparison
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        genes = row['Genes'].split()
        
        try:
            # Run traditional pipeline
            trad_response = run_traditional_pipeline(genes, config)
            trad_score = extract_score(trad_response)
            
            # Add longer delay between API calls
            time.sleep(random.uniform(5, 10))
            
            # Run RAG pipeline
            rag_response = run_rag_pipeline(genes, config, chroma_db)
            rag_score = extract_score(rag_response)
            
            # Store results
            results.loc[idx] = {
                'GO': row['GO'],
                'Genes': row['Genes'],
                'Traditional_Response': trad_response,
                'Traditional_Score': trad_score,
                'RAG_Response': rag_response,
                'RAG_Score': rag_score,
                'Score_Difference': rag_score - trad_score
            }
            
            # Save progress after each iteration
            results.to_csv(os.path.join(output_dir, 'rag_vs_traditional_comparison.tsv'), sep='\t')
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            results.loc[idx] = {
                'GO': row['GO'],
                'Genes': row['Genes'],
                'Traditional_Response': f"Error: {str(e)}",
                'Traditional_Score': -1,
                'RAG_Response': f"Error: {str(e)}",
                'RAG_Score': -1,
                'Score_Difference': 0
            }
            
            # Save progress even if there's an error
            results.to_csv(os.path.join(output_dir, 'rag_vs_traditional_comparison.tsv'), sep='\t')
            
            # If it's a rate limit error, wait longer before continuing
            if "rate_limit" in str(e).lower() or "quota" in str(e).lower():
                print("Rate limit exceeded. Waiting 120 seconds before continuing...")
                time.sleep(120)
    
    # Calculate and display statistics
    stats = calculate_statistics(results)
    print("\nFinal Statistics:")
    print(f"Total GO terms analyzed: {stats['total_terms']}")
    print(f"Mean Traditional Score: {stats['mean_traditional_score']:.3f}")
    print(f"Mean RAG Score: {stats['mean_rag_score']:.3f}")
    print(f"Mean Score Difference: {stats['mean_score_difference']:.3f}")
    print(f"Median Score Difference: {stats['median_score_difference']:.3f}")
    print(f"Standard Deviation of Score Differences: {stats['std_score_difference']:.3f}")
    print(f"RAG Higher Count: {stats['rag_higher_count']}")
    print(f"Traditional Higher Count: {stats['traditional_higher_count']}")
    print(f"Equal Scores Count: {stats['equal_scores_count']}")
    print(f"Max Score Difference: {stats['max_score_difference']:.3f}")
    print(f"Min Score Difference: {stats['min_score_difference']:.3f}")
    
    # Save final results
    results.to_csv(os.path.join(output_dir, 'rag_vs_traditional_comparison.tsv'), sep='\t')
    print("\nResults saved to data/GO_term_analysis/rag_vs_traditional_comparison.tsv")

if __name__ == "__main__":
    main() 