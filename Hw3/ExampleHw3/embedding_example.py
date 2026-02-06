"""
OpenAI Embedding API Example

What this does:
    Sends a short text string to an embedding model and prints basic details
    about the returned vector.

What you'll need:
    - openai: Library to connect to OpenAI's API

What you'll learn:
    - How to request embeddings from the API
    - How to inspect the embedding length and value types
"""

from openai import OpenAI

client = OpenAI(api_key='../../.env')
response = client.embeddings.create(
    input="CSC 4700 is really awesome!",
    model="text-embedding-3-small"
)
print(len(response.data[0].embedding))
print(type(response.data[0].embedding[0]))