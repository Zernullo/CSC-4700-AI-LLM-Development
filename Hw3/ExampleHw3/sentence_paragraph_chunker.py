"""
Sentence and Paragraph Chunking Helpers

What this does:
    Provides simple functions to chunk text by sentence or by paragraph,
    with maximum length controls for each chunk.

What you'll learn:
    - How to split text using basic regex sentence boundaries
    - How to cap chunk size by character count
"""

import re
from typing import List


def chunk_by_sentences(text: str, max_chars: int = 500) -> List[str]:
    """
    Splits text into sentence-level chunks, optionally grouping multiple sentences until max_chars is reached.

    :param text: he input text to split.
    :param max_chars: Maximum number of characters per chunk.
    :return: A list of sentence-based chunks.
    """
    # Basic sentence splitting using regex
    sentences = re.split(r'(?<=[.!?]) +', text.strip())

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunk_by_paragraphs(text: str, max_chars: int = 1000) -> List[str]:
    """
    Splits text into paragraph-level chunks, optionally splitting large paragraphs if they exceed max_chars.

    :param text: The input text to split.
    :param max_chars: Maximum number of characters per chunk.
    :return: A list of paragraph-based chunks.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
        else:
            # Split overly long paragraphs into fixed-size chunks
            for i in range(0, len(paragraph), max_chars):
                chunks.append(paragraph[i:i + max_chars])

    return chunks
