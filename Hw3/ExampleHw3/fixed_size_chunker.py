"""
Fixed-Size Chunking Example

What this does:
    Splits a document into fixed-length character chunks, with optional overlap
    between consecutive chunks.

What you'll learn:
    - How fixed-size chunking works
    - How overlap affects the chunk boundaries
"""

def fixed_size_chunks(text: str, chunk_size: int = 512, overlap: int = 0):
    """
    Splits a string into fixed-size character chunks.
    :param text: The input string to split.
    :param chunk_size: Maximum number of characters per chunk (default=512).
    :param overlap: Number of overlapping characters between consecutive chunks (default=0).
    :return: A list of text chunks.
    """
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk_size")
    step = chunk_size - overlap
    return [text[i:i + chunk_size] for i in range(0, len(text), step)]


# Example usage
doc = "This is a very long document string that we want to break into smaller chunks for retrieval purposes."
chunks = fixed_size_chunks(doc, chunk_size=20, overlap=5)
for i, c in enumerate(chunks, 1):
    print(f"Chunk {i}: {c}")
