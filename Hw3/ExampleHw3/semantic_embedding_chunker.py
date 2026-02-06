"""
Semantic Embedding Chunker Example

What this does:
    Splits text into chunks by embedding each sentence and creating a split
    whenever adjacent sentences are dissimilar enough.

What you'll learn:
    - How to compute sentence embeddings
    - How cosine similarity can drive semantic boundaries
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
import re
from typing import List


# set your embeddings model here
EMBED_MODEL = "text-embedding-3-small"   # or "text-embedding-3-large"

# Load API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D numpy arrays.

    :param v1: numpy array 1
    :param v2: numpy array 2
    :return: cosine similarity
    """
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.dot(v1, v2) / denom) if denom else 0.0


def embed_sentences(sentences: List[str], model=EMBED_MODEL):
    """
    Embeds a list of sentences using an OpenAI embedding model

    :param sentences:
    :param model:
    :return: a list of embedding vectors (list[float]) for given sentences
    """
    resp = client.embeddings.create(model=model, input=sentences)
    return [np.array(item.embedding, dtype=np.float32) for item in resp.data]


def semantic_split(text: str, threshold: float = 0.8, model=EMBED_MODEL) -> List[str]:
    """
    Split `text` into semantic chunks by comparing adjacent sentence embeddings. A new chunk starts when cosine
        similarity falls below `threshold`.

    :param text: The text to split
    :param threshold: the threshold below which a split point is declared (defauly 0.8)
    :param model: the OpenAI embedding model to use
    :return: a list of text chunks
    """
    # Simple sentence split; swap for spaCy/BLINGFIRE if you want stronger segmentation
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]
    if not sentences:
        return []

    embs = embed_sentences(sentences, model=model)

    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_similarity(embs[i-1], embs[i])
        if sim < threshold:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current:
        chunks.append(" ".join(current))
    return chunks


# Example usage
if __name__ == "__main__":
    doc = (
        "The company reported record profits in Q1 2025. "
        "Revenue increased by 15% year over year. "
        "Meanwhile, the CEO announced a new sustainability initiative. "
        "The program will focus on reducing carbon emissions across all operations. "
        "Future plans also include investments in renewable energy."
    )
    for i, ch in enumerate(semantic_split(doc, threshold=0.8), 1):
        print(f"--- Chunk {i} ---\n{ch}\n")
