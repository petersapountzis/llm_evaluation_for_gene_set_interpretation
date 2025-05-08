import pandas as pd
import json
from utils.chroma_utils import ChromaGO
from utils.openai_query import openai_chat
from utils.prompt_factory import make_user_prompt_with_score
import numpy as np
from typing import Dict, Any

def extract_score(response: str) -> float:
    """Extract confidence score from response."""
    try:
        # First try to find score in format (0.XX) after "Process:"
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
        'rag_higher_count': (results_df['Score_Difference'] > 0).sum(),
        'traditional_higher_count': (results_df['Score_Difference'] < 0).sum(),
        'equal_scores_count': (results_df['Score_Difference'] == 0).sum(),
        'max_score_difference': results_df['Score_Difference'].max(),
        'min_score_difference': results_df['Score_Difference'].min(),
        'std_score_difference': results_df['Score_Difference'].std()
    }
    
    return stats

def test_new_go_term(go_term, genes, config_path='jsonFiles/toyexample.json'):
    """
    Test the RAG system with a new GO term and its genes
    
    Args:
        go_term (str): GO term identifier
        genes (list): List of genes associated with the GO term
        config_path (str): Path to configuration file
    """
    # Initialize ChromaGO
    chroma_go = ChromaGO()
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get relevant context from ChromaDB
    print(f"\nRetrieving relevant context for {go_term}...")
    context = chroma_go.get_context_for_llm(genes)
    
    # Create prompts
    print("\nGenerating prompts...")
    traditional_prompt = make_user_prompt_with_score(genes=genes)
    
    # Create RAG prompt by combining context with the traditional prompt
    rag_prompt = f"""Context from GO database:
{context}

{traditional_prompt}"""
    
    # Run traditional analysis
    print("\nRunning traditional analysis...")
    traditional_response, _ = openai_chat(
        context=config['CONTEXT'],
        prompt=traditional_prompt,
        model=config['GPT_MODEL'],
        temperature=config['TEMP'],
        max_tokens=config['MAX_TOKENS'],
        rate_per_token=config['RATE_PER_TOKEN'],
        LOG_FILE=config['LOG_NAME'],
        DOLLAR_LIMIT=config['DOLLAR_LIMIT']
    )
    
    # Run RAG analysis
    print("\nRunning RAG analysis...")
    rag_response, _ = openai_chat(
        context=config['CONTEXT'],
        prompt=rag_prompt,
        model=config['GPT_MODEL'],
        temperature=config['TEMP'],
        max_tokens=config['MAX_TOKENS'],
        rate_per_token=config['RATE_PER_TOKEN'],
        LOG_FILE=config['LOG_NAME'],
        DOLLAR_LIMIT=config['DOLLAR_LIMIT']
    )
    
    # Display results
    print("\nResults:")
    print(f"\nGO Term: {go_term}")
    print(f"Genes: {', '.join(genes)}")
    print("\nTraditional Analysis:")
    print(traditional_response)
    print("\nRAG Analysis:")
    print(rag_response)
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'GO': [go_term],
        'Genes': [' '.join(genes)],
        'Traditional_Response': [traditional_response],
        'RAG_Response': [rag_response]
    })
    
    # Append to existing results file or create new one
    try:
        existing_df = pd.read_csv('data/GO_term_analysis/rag_vs_traditional_comparison.tsv', sep='\t')
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
    except FileNotFoundError:
        pass
    
    # Calculate and display statistics
    stats = calculate_statistics(results_df)
    print("\nSummary Statistics:")
    print(f"Total GO terms analyzed: {stats['total_terms']}")
    print(f"Mean Traditional Score: {stats['mean_traditional_score']:.3f}")
    print(f"Mean RAG Score: {stats['mean_rag_score']:.3f}")
    print(f"Mean Score Difference: {stats['mean_score_difference']:.3f}")
    print(f"RAG Higher Count: {stats['rag_higher_count']}")
    print(f"Traditional Higher Count: {stats['traditional_higher_count']}")
    print(f"Equal Scores Count: {stats['equal_scores_count']}")
    print(f"Max Score Difference: {stats['max_score_difference']:.3f}")
    print(f"Min Score Difference: {stats['min_score_difference']:.3f}")
    print(f"Standard Deviation of Score Differences: {stats['std_score_difference']:.3f}")
    
    # Save updated results
    results_df.to_csv('data/GO_term_analysis/rag_vs_traditional_comparison.tsv', sep='\t', index=False)
    print("\nResults saved to data/GO_term_analysis/rag_vs_traditional_comparison.tsv")

def test_from_csv(csv_path, config_path='jsonFiles/toyexample.json'):
    """
    Test multiple GO terms from a CSV file
    
    Args:
        csv_path (str): Path to CSV file with GO terms
        config_path (str): Path to configuration file
    """
    # Load CSV file
    df = pd.read_csv(csv_path)
    
    # Process each row
    for _, row in df.iterrows():
        print(f"\nProcessing GO term: {row['GO']}")
        genes = row['Genes'].split()
        test_new_go_term(row['GO'], genes, config_path)
    
    # Calculate final statistics after processing all terms
    try:
        results_df = pd.read_csv('data/GO_term_analysis/rag_vs_traditional_comparison.tsv', sep='\t')
        stats = calculate_statistics(results_df)
        print("\nFinal Summary Statistics:")
        print(f"Total GO terms analyzed: {stats['total_terms']}")
        print(f"Mean Traditional Score: {stats['mean_traditional_score']:.3f}")
        print(f"Mean RAG Score: {stats['mean_rag_score']:.3f}")
        print(f"Mean Score Difference: {stats['mean_score_difference']:.3f}")
        print(f"RAG Higher Count: {stats['rag_higher_count']}")
        print(f"Traditional Higher Count: {stats['traditional_higher_count']}")
        print(f"Equal Scores Count: {stats['equal_scores_count']}")
        print(f"Max Score Difference: {stats['max_score_difference']:.3f}")
        print(f"Min Score Difference: {stats['min_score_difference']:.3f}")
        print(f"Standard Deviation of Score Differences: {stats['std_score_difference']:.3f}")
    except FileNotFoundError:
        print("No results file found to calculate final statistics.")

if __name__ == "__main__":
    # Example usage
    print("Choose an option:")
    print("1. Test with a single GO term")
    print("2. Test with a CSV file")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        go_term = input("Enter GO term (e.g., GO:0000001): ")
        genes = input("Enter genes separated by spaces: ").split()
        test_new_go_term(go_term, genes)
    elif choice == "2":
        csv_path = input("Enter path to CSV file: ")
        test_from_csv(csv_path)
    else:
        print("Invalid choice") 