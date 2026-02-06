"""
OpenRouter API Completion Example

What this does:
    Shows how to use OpenRouter to access many different AI models (not just OpenAI).
    OpenRouter is a service that gives you one API to use models from OpenAI, Anthropic,
    Google, Meta, and other companies. You use the same OpenAI library, just with a
    different URL and API key.

What you'll need:
    - openai: Library to connect to APIs (works with OpenRouter too)
    - dotenv: Loads your API key from a .env file
    - os: Reads environment variables (like your API key)

What you'll learn:
    - How to change the base URL to connect to OpenRouter instead of OpenAI
    - How to access different AI models (like Qwen from Alibaba)
    - How the code looks almost identical to using OpenAI's API
    - Why you might want to use multiple AI providers
"""

from openai import OpenAI
from dotenv import load_dotenv
import os


# load environmental variables
load_dotenv('../../.env')

# establish client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# use GPT-40-mini to answer a simple question in a certain persona
response = client.chat.completions.create(
    model='qwen/qwen3-max',
    messages=[
        {"role": "system", "content": "You are a patient and helpful AI assistant named BobGPT. You speak like a pirate."},
        {"role": "user", "content": "How much wood could a woodchuck chuck if a woodchuck could chuck wood?"}
    ],
    temperature=0.7,
    top_p=1,
    max_tokens=250,
    n=1
)

# print the results
print("Model's Response:")
print('\t', response.choices[0].message.content)
