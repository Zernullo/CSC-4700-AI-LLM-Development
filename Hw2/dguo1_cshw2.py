"""
Author: Daniel Guo
CSC 4700 Homework 2: LLMs and Question Answering
Instructor: Dr. Keith Mills

This program uses OpenAI's GPT-5-nano and OpenRouter's Qwen3-8b models to answer
questions from the SQuAD dataset. It demonstrates how to interact with different
LLM APIs, handle batch processing, and save the results in JSON format.
Allowed libraries:
- openai
- dotenv
- pydantic
- tiktoken
- os
- json

References:
Google and ChatGPT were used to clarify syntax, standard library usage,
and API interactions.
"""

from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from typing import List
import tiktoken
import os
import json

# Load the SQuAD dev set
with open("dev-v2.0.json", "r", encoding="utf-8") as f:
    squad_data = json.load(f)

questions = [] 

# Loop through articles & paragraphs
for article in squad_data['data']:
    for paragraph in article['paragraphs']:
        for qa in paragraph['qas']:
            if not qa['is_impossible']:
                questions.append({
                    "question": qa['question'],
                    "answers": [ans['text'] for ans in qa['answers']]
                })
            if len(questions) >= 500:
                break
        if len(questions) >= 500:
            break
    if len(questions) >= 500:
        break

print(f"Collected {len(questions)} questions.")


# load environmental variables
load_dotenv('../../.env') 

# Standard OpenAI API client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# OpenRouter API client
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

"""
Use GPT-5-nano to answer a simple question in a certain persona
Batch processing to handle 3 batches of questions at a time
"""
min_answer = []
def minimal_reasoning(questions):
    for batch in range(0, len(questions), 3): 

        # Actually get the batch of questions
        batch_questions = questions[batch:batch+3]
        
        for q in batch_questions:
            response = openai_client.chat.completions.create(
                model='gpt-5-nano',
                messages=[
                    {"role": "system", "content": "Answer briefly. Do not explain your reasoning."},
                    {"role": "user", "content": q["question"]}
                ],
                temperature=0.7,
                top_p=1,
                max_tokens=250,
                n=1
            )   

            answer = response.choices[0].message.content
            min_answer.append({
                "question": q['question'], 
                "answer": answer, 
                "ground_truth": q['answers']
            })
        # Progress indicator
        if (q + 3) % 30 == 0:
            print(f"Processed {min(q + 3, len(questions))}/{len(questions)} questions")
    
    # Save the minimal reasoning answers to a JSON file
    filename = f"gpt-5-nano-hw2.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(min_answer, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(min_answer)} GPT-5-nano answers to {filename}")

"""
Use Qwen3-8b to answer a simple question in a certain persona
Batch processing to handle questions one by one
"""
sequential_answer = []
def sequentially(questions):
    for i, q in enumerate(questions):
        response = openrouter_client.chat.completions.create(
            model='qwen/qwen3-8b',
            messages=[
                {"role": "system", "content": "Answer briefly. Do not explain your reasoning."},
                {"role": "user", "content": q["question"]}
            ],
            temperature=0.7,
            top_p=1,
            max_tokens=250,
            n=1
        )   

        # Append the question and answer as a dictionary to the sequential_answer list
        sequential_answer.append({
            "question": q['question'], 
            "answer": response.choices[0].message.content, 
            "ground_truth": q['answers']
        }) 
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(questions)} questions")
    
    # Save the sequential answers to a JSON file
    filename = f"qwen3-8b-hw2.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(sequential_answer, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(sequential_answer)} Qwen3-8b answers to {filename}")

# Simple menu to select options
while True:
    choice = input(
        "Enter 1-4:\n"
        "1. Minimal Reasoning\n"
        "2. Option 2\n"
        "3. Option 3\n"
        "4. Quit\n"
    )
    if choice == "1":
        minimal_reasoning(questions)
    elif choice == "2":
        sequentially(questions)
    elif choice == "3":
        pass
    elif choice == "4":
        print("Exiting program.")
        exit()
# Define a simple JSON Schema for structured output
# schema = {
#     "question": "question_extraction",
#     "schema": {
#         "type": "object",
#         "properties": {
#             "question": { "type": "string" },
#             "age":  { "type": "integer", "minimum": 0 }
#         },
#         "required": ["question", "age"],
#         "additionalProperties": False
#     }
# }

# messages = [
#     {"role": "system", "content": "You extract fields and output ONLY JSON that satisfies the provided schema."},
#     {"role": "user",   "content": "From this sentence, extract the person's name and age: 'Alice is 29 years old.'"}
# ]

# # Structured Outputs (JSON Schema / "struct mode")
# resp = client.chat.completions.create(
#     model="gpt-5-nano",
#     messages=messages,
#     temperature=0.0,
#     response_format={"type": "json_schema", "json_schema": schema}
# )