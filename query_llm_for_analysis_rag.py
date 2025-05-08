import argparse
import json
import pandas as pd
import numpy as np
from utils.openai_query import openai_chat
from utils.prompt_factory import make_user_prompt_with_score
from utils.chroma_utils import ChromaGO
import os
import sys

def validate_path(path, description):
    """Validate that a path exists."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} not found: {path}")
    return path

def main(df, chroma_db, config, column_prefix, gene_column, gene_sep=' ', output_file=None):
    """Main function to process gene sets using RAG-enhanced LLM analysis."""
    
    # Initialize Chroma database if not already done
    if not os.path.exists("./chroma_db"):
        chroma_db.load_go_terms("data/go_terms.csv")
    
    for idx, row in df.iterrows():
        # Skip if already processed
        if pd.notna(df.loc[idx, f'{column_prefix} Name']):
            continue
            
        # Get gene list
        genes = row[gene_column].split(gene_sep)
        
        # Get relevant context from Chroma
        context = chroma_db.get_context_for_llm(genes)
        
        # Format context for LLM
        llm_context = f"""You are an efficient and insightful assistant to a molecular biologist.
You should give the true answer that is supported by the references. If you do not have a clear answer, you will respond with "Unknown".

Important context for these genes can be found here:
{context}
"""
        
        # Create prompt
        prompt = make_user_prompt_with_score(genes=genes)
        
        try:
            # Query LLM
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
            
            # Parse response
            try:
                # Extract name and score from response
                lines = response_text.split('\n')
                if len(lines) > 0 and ': ' in lines[0]:
                    name_part = lines[0].split(': ')[1]
                    if '(' in name_part and ')' in name_part:
                        name = name_part.split(' (')[0]
                        score = float(name_part.split('(')[1].split(')')[0])
                    else:
                        name = name_part
                        score = 0.0
                else:
                    name = "Unknown"
                    score = 0.0
                
                # Update dataframe
                df.loc[idx, f'{column_prefix} Name'] = name
                df.loc[idx, f'{column_prefix} Analysis'] = response_text
                df.loc[idx, f'{column_prefix} Score'] = score
                
            except Exception as e:
                print(f"Error parsing response for {idx}: {e}")
                print(f"Response text: {response_text}")
                df.loc[idx, f'{column_prefix} Name'] = "Error"
                df.loc[idx, f'{column_prefix} Analysis'] = response_text
                df.loc[idx, f'{column_prefix} Score'] = -1
                
        except Exception as e:
            print(f"Error querying LLM for {idx}: {e}")
            df.loc[idx, f'{column_prefix} Name'] = "Error"
            df.loc[idx, f'{column_prefix} Analysis'] = f"Error: {str(e)}"
            df.loc[idx, f'{column_prefix} Score'] = -1
            
        # Save progress periodically
        if output_file and idx % 10 == 0:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(f"{output_file}_progress.tsv", sep='\t')
            
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run LLM analysis with RAG')
    parser.add_argument('--config', required=True, help='Path to config JSON file')
    parser.add_argument('--initialize', action='store_true', help='Initialize output columns')
    parser.add_argument('--input', required=True, help='Input file path')
    parser.add_argument('--input_sep', default='\t', help='Input file separator')
    parser.add_argument('--set_index', required=True, help='Column to use as index')
    parser.add_argument('--gene_column', required=True, help='Column containing gene lists')
    parser.add_argument('--gene_sep', default=' ', help='Separator for genes in gene_column')
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--end', type=int, help='End index')
    parser.add_argument('--output_file', required=True, help='Output file path')
    
    args = parser.parse_args()
    
    # Validate paths
    config_path = validate_path(args.config, "Config file")
    input_path = validate_path(args.input, "Input file")
    
    # Load config
    with open(config_path) as f:
        config = json.load(f)
        
    # Load data
    df = pd.read_csv(input_path, sep=args.input_sep, index_col=args.set_index)
    
    # Initialize Chroma database
    chroma_db = ChromaGO()
    
    # Process data
    if args.end:
        df = df.iloc[args.start:args.end]
        
    # Initialize columns if needed
    if args.initialize:
        column_prefix = config['GPT_MODEL'].replace('-', '_')
        df[f'{column_prefix} Name'] = None
        df[f'{column_prefix} Analysis'] = None
        df[f'{column_prefix} Score'] = -np.inf
        
    # Run analysis
    df = main(df, chroma_db, config, column_prefix, args.gene_column, args.gene_sep, args.output_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Save results
    df.to_csv(f"{args.output_file}.tsv", sep='\t')
    print("Done") 