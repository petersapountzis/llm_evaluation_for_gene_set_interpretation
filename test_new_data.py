import pandas as pd
import json
from utils.chroma_utils import get_relevant_context
from utils.openai_query import openai_chat
from utils.prompt_factory import make_user_prompt_with_score

def test_new_gene_set(gene_list, config_path='jsonFiles/toyexample.json'):
    """
    Test the RAG system with a new gene list
    
    Args:
        gene_list (list): List of genes to analyze
        config_path (str): Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get relevant context from ChromaDB
    print("\nRetrieving relevant context from ChromaDB...")
    context = get_relevant_context(gene_list)
    
    # Create prompts
    print("\nGenerating prompts...")
    traditional_prompt = make_user_prompt_with_score(genes=gene_list)
    rag_prompt = make_user_prompt_with_score(genes=gene_list, context=context)
    
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
    print(f"\nGenes: {', '.join(gene_list)}")
    print("\nTraditional Analysis:")
    print(traditional_response)
    print("\nRAG Analysis:")
    print(rag_response)

def test_from_csv(csv_path, config_path='jsonFiles/toyexample.json'):
    """
    Test multiple gene sets from a CSV file
    
    Args:
        csv_path (str): Path to CSV file with gene sets
        config_path (str): Path to configuration file
    """
    # Load CSV file
    df = pd.read_csv(csv_path)
    
    # Process each row
    for _, row in df.iterrows():
        print(f"\nProcessing GO term: {row['GO']}")
        gene_list = row['Genes'].split()
        test_new_gene_set(gene_list, config_path)

if __name__ == "__main__":
    # Example usage
    print("Choose an option:")
    print("1. Test with a single gene list")
    print("2. Test with a CSV file")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        genes = input("Enter genes separated by spaces: ").split()
        test_new_gene_set(genes)
    elif choice == "2":
        csv_path = input("Enter path to CSV file: ")
        test_from_csv(csv_path)
    else:
        print("Invalid choice") 