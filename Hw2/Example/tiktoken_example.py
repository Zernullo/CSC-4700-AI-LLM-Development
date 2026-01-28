"""
Tiktoken Token Counting Example

What this does:
    Shows how to count tokens in your text before sending it to the API.
    This is important because OpenAI charges you based on tokens, and models
    have limits on how many tokens they can process. Knowing the count helps
    you estimate costs and avoid hitting limits.

What you'll need:
    - tiktoken: OpenAI's official tool for counting tokens
        (Tokens are pieces of words that the AI model understands)

What you'll learn:
    - How to count tokens for different GPT models
    - How to get the right token counter for your specific model
    - What to do if your model isn't recognized (fallback method)
    - How to create a reusable function for token counting
"""

import tiktoken

def count_tokens(text: str, model_name: str) -> int:
    """
    Counts the number of tokens in a text string for a given model.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback for models not explicitly in tiktoken's registry
        encoding = tiktoken.get_encoding("cl100k_base") 
    
    tokens = encoding.encode(text)
    return len(tokens)

# Example usage:
if __name__ == "__main__":
    prompt_text = "Hello, world! How are you today?"
    model = "gpt-4o" # or "gpt-3.5-turbo", etc.
    token_count = count_tokens(prompt_text, model)
    print(f"The text has {token_count} tokens.")
