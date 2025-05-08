from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import pandas as pd
from typing import List, Dict, Any
import time
import os

class ChromaGO:
    def __init__(self, db_path: str = "./chroma_db"):
        """Initialize Chroma database for GO terms."""
        self.client = PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection("go_terms_sorted")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
    def load_go_terms(self, go_terms_file: str, max_retries: int = 3) -> None:
        """Load GO terms from CSV file into Chroma database."""
        if not os.path.exists(go_terms_file):
            raise FileNotFoundError(f"GO terms file not found: {go_terms_file}")
            
        # Load GO terms
        df = pd.read_csv(go_terms_file)  # columns: GO, Term_Description, Genes
        
        # Sort gene lists alphabetically
        def sort_genes(gene_str):
            genes = gene_str.split()
            return " ".join(sorted(genes))
            
        df["Sorted_Genes"] = df["Genes"].apply(sort_genes)
        
        # Format text for embedding
        df["text"] = df.apply(
            lambda row: f"{row['GO']}: {row['Term_Description']} | Genes: {row['Sorted_Genes']}",
            axis=1
        )
        
        # Embed texts
        print("Embedding GO terms...")
        df["embedding"] = df["text"].apply(lambda x: self.model.encode(x).tolist())
        
        # Upload to Chroma if collection is empty
        if self.collection.count() == 0:
            print("Uploading GO terms to Chroma...")
            for attempt in range(max_retries):
                try:
                    self.collection.add(
                        documents=df["text"].tolist(),
                        embeddings=df["embedding"].tolist(),
                        ids=df["GO"].astype(str).tolist()
                    )
                    print(f"Successfully uploaded {len(df)} GO terms to Chroma")
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise Exception(f"Failed to upload GO terms after {max_retries} attempts: {e}")
                    print(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
        else:
            print(f"{self.collection.count()} items already exist in Chroma.")
            
    def query_genes(self, gene_list: List[str], n_results: int = 5, max_retries: int = 3) -> Dict[str, Any]:
        """Query Chroma database for relevant GO terms based on gene list."""
        if not gene_list:
            raise ValueError("Gene list cannot be empty")
            
        # Format query
        query = " ".join(sorted(gene_list))
        
        # Embed query
        query_embedding = self.model.encode(query).tolist()
        
        # Search Chroma with retries
        for attempt in range(max_retries):
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
                return {
                    'documents': results['documents'][0],
                    'ids': results['ids'][0],
                    'distances': results['distances'][0]
                }
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to query Chroma after {max_retries} attempts: {e}")
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
                
    def get_context_for_llm(self, gene_list: List[str], n_results: int = 5) -> str:
        """Get formatted context for LLM from Chroma results."""
        try:
            results = self.query_genes(gene_list, n_results)
            return "\n".join(results['documents'])
        except Exception as e:
            print(f"Error getting context for genes {gene_list}: {e}")
            return "Error retrieving context from database." 