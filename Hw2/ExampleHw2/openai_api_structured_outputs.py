"""
OpenAI API Structured Outputs (Manual JSON Schema) Example

What this does:
    Shows how to get structured JSON data from GPT by manually writing a JSON schema.
    This is an alternative to using Pydantic models - you define the exact format
    you want using a schema, and the API guarantees it will match.

What you'll need:
    - openai: Library to connect to OpenAI's API
    - dotenv: Loads your API key from a .env file
    - os: Reads environment variables (like your API key)
    - json: Converts the JSON response into Python dictionaries

What you'll learn:
    - How to write a JSON schema by hand (with nested objects and arrays)
    - How to force the API to always return data in your exact format
    - How to handle complex data structures (like a list of emperors)
    - How to parse JSON and access the data using dictionary keys
    - How to calculate the cost based on token usage
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import json


# load environmental variables
load_dotenv('../../.env')

# establish client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# use GPT-40-mini to answer a simple question in a certain persona
response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[
        {"role": "system", "content": "You are an expert in ancient Roman history."},
        {"role": "user", "content": "Generate a complete list of emperors, beginning with the fall of the republic."}
    ],
    reasoning_effort="low",
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "roman_emperors",
            "schema": {
                "type": "object",
                "properties": {
                    "emperors": {
                        "type": "array",
                        "description": "List of each emperor",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Name of the emperor"},
                                "first_year": {"type": "integer", "description": "First year of reign (use negative for BC)"},
                                "last_year": {"type": "integer", "description": "Last year of reign (use negative for BC)"},
                                "why_ended": {"type": "string", "description": "A concise reason for why their reign ended"}
                            },
                            "required": ["name", "first_year", "last_year", "why_ended"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["emperors"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
)
json_object = json.loads(response.choices[0].message.content)
print(json_object)
# print the results
print("Model's Formatted Response:")
for emperor in json_object['emperors']:
    print(emperor['name'])
    print(f"\t First Year: {emperor['first_year']}")
    print(f"\t Last Year:  {emperor['last_year']}")
    print(f"\t Why Ended:  {emperor['why_ended']}")

print()
print(f"Input Tokens:  {response.usage.prompt_tokens}")
print(f"Output Tokens: {response.usage.completion_tokens}")
print(f"Cost: ${response.usage.prompt_tokens * 0.05 / 1E6 + response.usage.completion_tokens * 0.4 / 1E6}")
