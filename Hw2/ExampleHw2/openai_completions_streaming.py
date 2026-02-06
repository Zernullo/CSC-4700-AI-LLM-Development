"""
OpenAI API Streaming Completions Example

What this does:
    Shows how to get responses from GPT in real-time as they're being generated.
    Instead of waiting for the complete answer, you see words appear one by one,
    just like when you use ChatGPT - this makes the app feel more responsive.

What you'll need:
    - openai: Library to connect to OpenAI's API
    - dotenv: Loads your API key from a .env file
    - os: Reads environment variables (like your API key)

What you'll learn:
    - How to enable streaming mode to get responses in real-time
    - How to process each chunk of text as it arrives
    - How to display partial results immediately
    - Why this is useful for chatbots and interactive apps
"""

from openai import OpenAI
from dotenv import load_dotenv
import os


# load environmental variables
load_dotenv('../../.env')

# establish client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# use GPT-40-mini to answer a simple question in a certain persona
stream = client.chat.completions.create(
    model='gpt-5-nano',
    messages=[
        {"role": "system", "content": "You are a patient and helpful AI assistant named BobGPT."},
        {"role": "user", "content": "Give me a list of every one of the Roman emperors."}
    ],
    reasoning_effort="minimal",
    stream=True
)

print("Model's Response:")
for chunk in stream:
    chunk_content = chunk.choices[0].delta.content
    if chunk_content is not None:
        print(chunk_content, end="")

