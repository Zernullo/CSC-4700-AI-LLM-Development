"""
OpenAI API Completion Example

What this does:
    Shows how to send a basic chat request to OpenAI's GPT model and get a response.
    This example makes a simple API call, lets you customize the assistant's behavior,
    and calculates how much the API call costs.

What you'll need:
    - openai: Library to connect to OpenAI's API
    - dotenv: Loads your API key from a .env file
    - os: Reads environment variables (like your API key)

What you'll learn:
    - How to create a chat completion with a custom role (like a pirate assistant)
    - How to control response creativity (temperature) and length (max tokens)
    - How to track token usage and calculate the cost
    - How to get structured JSON output from the model
"""

from openai import OpenAI
from dotenv import load_dotenv
import os


# load environmental variables
load_dotenv('../../.env')

# establish client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# use GPT-40-mini to answer a simple question in a certain persona
response = client.chat.completions.create(
    model='gpt-4o-mini',
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
print()
print(f"Input Tokens:  {response.usage.prompt_tokens}")
print(f"Output Tokens: {response.usage.completion_tokens}")
print(f"Cost: ${response.usage.prompt_tokens * 0.15 / 1E6 + response.usage.completion_tokens * 0.6 / 1E6}")

# Define a simple JSON Schema for structured output
schema = {
    "name": "person_extraction",
    "schema": {
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age":  { "type": "integer", "minimum": 0 }
        },
        "required": ["name", "age"],
        "additionalProperties": False
    }
}

messages = [
    {"role": "system", "content": "You extract fields and output ONLY JSON that satisfies the provided schema."},
    {"role": "user",   "content": "From this sentence, extract the person's name and age: 'Alice is 29 years old.'"}
]

# Structured Outputs (JSON Schema / "struct mode")
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature=0.0,
    response_format={"type": "json_schema", "json_schema": schema}
)