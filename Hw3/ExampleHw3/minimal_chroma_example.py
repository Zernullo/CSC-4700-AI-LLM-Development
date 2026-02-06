"""
Minimal ChromaDB RAG Example

What this does:
    Creates a small persistent Chroma collection, inserts a few documents,
    and runs a filtered similarity query.

What you'll learn:
    - How to configure an embedding function
    - How to insert documents and query with metadata filters
"""

from chromadb.utils import embedding_functions
import chromadb, os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)
client = chromadb.PersistentClient(path="./kb")
col = client.get_or_create_collection("handbook", embedding_function=ef)

# Ingest some chunks (after your chunking step)
col.upsert(
    ids=["hbk-1","hbk-2","hbk-3"],
    documents=[
        "Employees must complete security training annually.",
        "VPN and MFA are required for remote access.",
        "Contractors must follow the same access policy."
    ],
    metadatas=[{"section":"training"},{"section":"access"},{"section":"access"}]
)

# Query with filters
question = "Do contractors need security training? What about remote access rules?"
res = col.query(query_texts=[question], n_results=3, where={"section":{"$in":["training","access"]}})
contexts = res["documents"][0]
print(contexts)
