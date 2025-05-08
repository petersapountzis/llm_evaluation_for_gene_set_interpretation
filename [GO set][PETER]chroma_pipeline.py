#%%
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

client = PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("go_terms")

model = SentenceTransformer("all-MiniLM-L6-v2")
#%% md
# ## Load and embed GO terms
#%%
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load your GO terms (this should already exist)
df = pd.read_csv("data/go_terms.csv")  # must contain 'id', 'name', 'definition'

# Format text for embedding
df["text"] = df.apply(lambda row: f"{row['GO']}: {row['Term_Description']} | Genes: {row['Genes']}", axis=1)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed texts#%%
# from chromadb import PersistentClient
# from sentence_transformers import SentenceTransformer
#
# client = PersistentClient(path="./chroma_db")
# collection = client.get_or_create_collection("go_terms")
#
# model = SentenceTransformer("all-MiniLM-L6-v2")
# #%% md
# # ## Load and embed GO terms
# #%%
# import pandas as pd
# from sentence_transformers import SentenceTransformer
#
# # Load your GO terms (this should already exist)
# df = pd.read_csv("data/go_terms.csv")  # must contain 'id', 'name', 'definition'
#
# # Format text for embedding
# df["text"] = df.apply(lambda row: f"{row['GO']}: {row['Term_Description']} | Genes: {row['Genes']}", axis=1)
#
# # Load embedding model
# model = SentenceTransformer("all-MiniLM-L6-v2")
#
# # Embed texts
# df["embedding"] = df["text"].apply(lambda x: model.encode(x).tolist())
# df.head(2)
# #%%
# # Upload only if collection is empty
# if collection.count() == 0:
#     print("Uploading GO terms to Chroma...")
#     collection.add(
#         documents=df["text"].tolist(),
#         embeddings=df["embedding"].tolist(),
#         ids=df["GO"].astype(str).tolist()
#     )
# else:
#     print(f"{collection.count()} items already exist in Chroma.")
# #%%
# # Define gene list (you can later make this dynamic)
# gene_list = ["TP53", "BAX", "CASP3"]
# query = f"What biological process is most likely shared by the genes: {', '.join(gene_list)}"
#
# # Embed query
# query_embedding = model.encode(query).tolist()
#
# # Search Chroma for top-K relevant GO entries
# results = collection.query(query_embeddings=[query_embedding], n_results=5)
#
# # Print the retrieved GO context
# top_context = "\n".join(results['documents'][0])
# print("Top GO context:\n", top_context)
# #%%
# import sys
# sys.path.append("utils")
#
# from openai_query import openai_chat
# from prompt_factory import make_user_prompt_with_score
# #%%
# # Context and prompt setup
# context = """You are an efficient and insightful assistant to a molecular biologist.
# You should give the true answer that is supported by the references. If you do not have a clear answer, you will respond with "Unknown".
#
# Important context for these genes can be found here:
# {top_context}
# """
#
# # prompt = f"""The following GO terms describe gene functions:
# #
# # {top_context}
# #
# # Given the gene list: TP53, BAX, CASP3, what biological process do they most likely share?
# # """
# gene_list = ["TP53", "BAX", "CASP3"]
# prompt = make_user_prompt_with_score(genes=gene_list)
#
# # Query params
# model = "gpt-3.5-turbo"
# temperature = 0
# max_tokens = 500
# rate_per_token = 0.0005
# LOG_FILE = "logs/test_openai_log.json"
# DOLLAR_LIMIT = 1.00
#
# #%%
# response_text, fingerprint = openai_chat(
#     context=context,
#     prompt=prompt,
#     model=model,
#     temperature=temperature,
#     max_tokens=max_tokens,
#     rate_per_token=rate_per_token,
#     LOG_FILE=LOG_FILE,
#     DOLLAR_LIMIT=DOLLAR_LIMIT,
#     seed=42
# )
#
# print("🔬 GPT Response:\n", response_text)
# #%%
df["embedding"] = df["text"].apply(lambda x: model.encode(x).tolist())
df.head(2)
#%%
# Upload only if collection is empty
if collection.count() == 0:
    print("Uploading GO terms to Chroma...")
    collection.add(
        documents=df["text"].tolist(),
        embeddings=df["embedding"].tolist(),
        ids=df["GO"].astype(str).tolist()
    )
else:
    print(f"{collection.count()} items already exist in Chroma.")
#%%
# Define gene list (you can later make this dynamic)
gene_list = ["TP53", "BAX", "CASP3"]
query = f"What biological process is most likely shared by the genes: {', '.join(gene_list)}"

# Embed query
query_embedding = model.encode(query).tolist()

# Search Chroma for top-K relevant GO entries
results = collection.query(query_embeddings=[query_embedding], n_results=5)

# Print the retrieved GO context
top_context = "\n".join(results['documents'][0])
print("Top GO context:\n", top_context)
#%%
import sys
sys.path.append("utils") 

from openai_query import openai_chat
from prompt_factory import make_user_prompt_with_score
#%%
# Context and prompt setup
context = """You are an efficient and insightful assistant to a molecular biologist.
You should give the true answer that is supported by the references. If you do not have a clear answer, you will respond with "Unknown".

Important context for these genes can be found here:
{top_context}
"""

# prompt = f"""The following GO terms describe gene functions:
# 
# {top_context}
# 
# Given the gene list: TP53, BAX, CASP3, what biological process do they most likely share?
# """
gene_list = ["TP53", "BAX", "CASP3"]
prompt = make_user_prompt_with_score(genes=gene_list)

# Query params
model = "gpt-3.5-turbo"
temperature = 0
max_tokens = 500
rate_per_token = 0.0005
LOG_FILE = "logs/test_openai_log.json"
DOLLAR_LIMIT = 1.00

#%%
response_text, fingerprint = openai_chat(
    context=context,
    prompt=prompt,
    model=model,
    temperature=temperature,
    max_tokens=max_tokens,
    rate_per_token=rate_per_token,
    LOG_FILE=LOG_FILE,
    DOLLAR_LIMIT=DOLLAR_LIMIT,
    seed=42
)

print("🔬 GPT Response:\n", response_text)
#%%
