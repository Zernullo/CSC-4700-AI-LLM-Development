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
- datetime

References:
Google and ChatGPT were used to clarify syntax, standard library usage,
and API interactions.
"""

from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os
import json
import time
import base64

# globally so it stays consistent for the whole session
CURRENT_DATE = datetime.now().strftime("%Y%m%d")

# ===========================================================================
# STEP 1: SETUP AND DATA LOADING
# ============================================================================

# Load the SQuAD dev set
with open("dev-v2.0.json", "r", encoding="utf-8") as f:
    squad_data = json.load(f)

questions = [] 

# Loop through articles & paragraphs
for article in squad_data['data']:
    for paragraph in article['paragraphs']:
        for qa in paragraph['qas']:
            # Skip impossible questions (those without answers)
            if not qa.get('is_impossible', False):
                # Extract unique answer texts (SQuAD has multiple annotator answers)
                answer_texts = [ans['text'] for ans in qa.get('answers', [])]
                # Only add if there are actual answers
                if answer_texts:
                    questions.append({
                        "question": qa['question'],
                        "answers": answer_texts,
                        "id": qa.get('id', f'q_{len(questions)}')
                    })
            if len(questions) >= 500:
                break
        if len(questions) >= 500:
            break
    if len(questions) >= 500:
        break

print(f"Collected {len(questions)} questions.")

# ============================================================================
# SETUP OPENAI AND OPENROUTER CLIENTS
# ============================================================================
# load environmental variables
load_dotenv('.env') 

# Standard OpenAI API client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# OpenRouter API client
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# ============================================================================
# QUESTION 2: GPT-5-nano using OpenAI Batch API
# ============================================================================
"""
Use GPT-5-nano to answer a simple question in a certain persona
"""
gpt_5_nano_answer = []
def gpt_5(questions, batch_size=100):
    global gpt_5_nano_answer
    gpt_5_nano_answer = []
    print("Starting GPT-5-nano batch processing...")
    batch_ids = []
    batch_number = 0
    batch_id_to_number = {}
    for i in range(0, len(questions), batch_size): 
        # Actually get the batch of questions
        batch_questions = questions[i:i+batch_size]
        batch_number += 1
        input_file = f"batch_{batch_number}_requests.jsonl"
        with open(input_file, 'w', encoding='utf-8') as f:
            for j, q in enumerate(batch_questions):
                request = {
                    "custom_id": f"batch{batch_number}-q{j}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-5-nano",
                        "messages": [
                            {"role": "system", "content": "Answer briefly, 1-5 words. Do not explain your reasoning. If ambiguous, provide the most common answer."},
                            {"role": "user", "content": q["question"]}
                        ],
                        "response_format": {"type": "text"},
                    }
                }
                f.write(json.dumps(request) + '\n')

        # Upload file
        batch_input_file = openai_client.files.create(
            file=open(input_file, "rb"),
            purpose="batch"
        )
        # Create batch job
        batch = openai_client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        batch_ids.append(batch.id)
        batch_id_to_number[batch.id] = batch_number
        print(f"  Batch {batch_number}: {batch.id} submitted ({len(batch_questions)} questions)")
    
    print(f"\n✓ All {len(batch_ids)} batch(es) submitted!")


    # Step 2: Wait for all batches to complete
    print(f"\nStep 2: Waiting for batches to complete...")
    completed_batches = []
    while len(completed_batches) < len(batch_ids):
        for batch_id in batch_ids:
            if batch_id in completed_batches:
                continue
            batch_status = openai_client.batches.retrieve(batch_id)
            if batch_status.status == "completed":
                completed_batches.append(batch_id)
                print(f"  Batch {batch_id_to_number[batch_id]} completed.")
        if len(completed_batches) < len(batch_ids):
            print("  Waiting for 1 minute before checking again...")
            time.sleep(60)
    print(f"\n✓ All batches finished!")

    # Step 3: Download and merge results
    print(f"\nStep 3: Downloading and merging results...")
    for batch_id in completed_batches:
        batch_obj = openai_client.batches.retrieve(batch_id)
        if batch_obj.status != "completed":
            continue

        # If there is an error file, fetch and print it
        if batch_obj.error_file_id:
            error_content = openai_client.files.content(batch_obj.error_file_id).text
            print(f"Batch {batch_id_to_number[batch_id]} errors:\n{error_content}")

        if not batch_obj.output_file_id:
            print(f"Batch {batch_id_to_number[batch_id]} has no output_file_id.")
            continue

        result_content = openai_client.files.content(batch_obj.output_file_id).text
        if not result_content.strip():
            print(f"Batch {batch_id_to_number[batch_id]} output is empty.")
            continue

        # Save batch results locally
        results_file = f"batch_{batch_id_to_number[batch_id]}_results.jsonl"
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(result_content)

        # Parse each line in results
        for line in result_content.splitlines():
            result = json.loads(line)
            if "response" not in result:
                continue
            custom_id = result['custom_id']
            batch_num, q_num = map(int, custom_id.replace('batch', '').split('-q'))
            original_question = questions[(batch_num - 1) * batch_size + q_num]
            choices = result['response']['body'].get('choices', [])
            if not choices:
                continue
            answer = choices[0]['message']['content']
            gpt_5_nano_answer.append({
                "question": original_question['question'],
                "answer": answer,
                "ground_truth": original_question['answers']
            })
    print(f"Collected {len(gpt_5_nano_answer)} answers from GPT-5-nano.")

    # Save the minimal reasoning answers to a JSON file
    filename = f"gpt-5-nano-{CURRENT_DATE}-hw2.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(gpt_5_nano_answer, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(gpt_5_nano_answer)} GPT-5-nano answers to {filename}")


# ============================================================================
# QUESTION 3: Qwen3-8b using OpenRouter API
# ============================================================================
"""
Use Qwen3-8b to answer a simple question in a certain persona
"""
qwen3_8b_answer = []
def qwen3_8b(questions):
    global qwen3_8b_answer
    qwen3_8b_answer = []
    print("Starting Qwen3-8b sequential processing...")
    for i, q in enumerate(questions):
        response = openrouter_client.chat.completions.create(
            model='qwen/qwen3-8b',
            messages=[
                {"role": "system", "content": "Answer briefly, 1-5 words. Do not explain your reasoning. If ambiguous, provide the most common answer."},
                {"role": "user", "content": q["question"]}
            ],
            temperature=0.3,  
            top_p=1,
            n=1
        )

        # Append the question and answer as a dictionary to the qwen3_8b_answer list
        qwen3_8b_answer.append({
            "question": q['question'], 
            "answer": response.choices[0].message.content or "",
            "ground_truth": q['answers']
        }) 
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(questions)} questions")
    
    # Save the sequential answers to a JSON file
    filename = f"qwen3-8b-{CURRENT_DATE}-hw2.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(qwen3_8b_answer, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(qwen3_8b_answer)} Qwen3-8b answers to {filename}")


# ============================================================================
# QUESTION 4: Scoring the Model Answers
# ============================================================================
"""
Use GPT-5-mini to score the answers from both models
"""
class ScoringResponse(BaseModel):
    explanation: str = Field(description="A short explanation of why the student's answer was correct or incorrect")
    score: bool = Field(description="true if the student's answer was correct, false if it was incorrect")

def scoring_result(model_name, result_file, batch_size=100):
    print(f"Starting scoring for {model_name}...")
    
    with open(result_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    total = len(results)
    completed = sum(1 for r in results if str(r.get("answer", "")).strip())
    completion_rate = (completed / total) * 100 if total else 0

    # Step 1: Create and submit batches
    print(f"\nStep 1: Creating batches of {batch_size} questions...")
    batch_ids = []
    batch_number = 0
    batch_id_to_number = {}
    
    for i in range(0, len(results), batch_size):
        batch_results = results[i:i+batch_size]
        batch_number += 1
        input_file = f"{model_name}_scoring_batch_{batch_number}_requests.jsonl"
        
        with open(input_file, "w", encoding="utf-8") as f:
            for j, result in enumerate(batch_results):
                global_index = i + j
                request = {
                    "custom_id": f"{model_name}-batch{batch_number}-score-{global_index}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-5-mini",
                        "messages": [
                            {
                                "role": "system",
                                "content":
                                "You are a teacher tasked with determining whether a student's answer to a question was correct, "
                                "based on a set of possible correct answers. You must only use the provided possible correct "
                                "answers to determine if the student's response was correct."
                            },
                            {
                                "role": "user",
                                "content":
                                f"Question: {result['question']}\n"
                                f"Student's Response: {result['answer']}\n"
                                f"Possible Correct Answers:\n{result['ground_truth']}\n"
                                f"Your response should only be a valid Json as shown below:\n"
                                "{\n"
                                "  \"explanation\" (str): A short explanation of why the student's answer was correct or incorrect.,\n"
                                "  \"score\" (bool): true if the student's answer was correct, false if it was incorrect\n"
                                "}\n"
                                "Your response:"
                            }
                        ],
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "ScoringResponse",
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "explanation": {"type": "string"},
                                        "score": {"type": "boolean"}
                                    },
                                    "required": ["explanation", "score"],
                                    "additionalProperties": False
                                }
                            }
                        },
                    }
                }
                f.write(json.dumps(request) + "\n")

        # Upload file
        batch_input_file = openai_client.files.create(
            file=open(input_file, "rb"),
            purpose="batch"
        )

        # Create batch job
        batch = openai_client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        batch_ids.append(batch.id)
        batch_id_to_number[batch.id] = batch_number
        print(f"  Batch {batch_number}: {batch.id} submitted ({len(batch_results)} questions)")

    print(f"\n✓ All {len(batch_ids)} batch(es) submitted!")

    # Step 2: Wait for all batches to complete
    print(f"\nStep 2: Waiting for batches to complete...")
    completed_batches = []
    while len(completed_batches) < len(batch_ids):
        for batch_id in batch_ids:
            if batch_id in completed_batches:
                continue
            batch_status = openai_client.batches.retrieve(batch_id)
            if batch_status.status == "completed":
                completed_batches.append(batch_id)
                print(f"  Batch {batch_id_to_number[batch_id]} completed.")
            elif batch_status.status in ("failed", "cancelled", "expired"):
                print(f"  Batch {batch_id_to_number[batch_id]} ended with status: {batch_status.status}")
                if batch_status.error_file_id:
                    err = openai_client.files.content(batch_status.error_file_id).text
                    print(f"  Batch {batch_id_to_number[batch_id]} errors:\n{err}")
                completed_batches.append(batch_id)  # Mark as done even if failed
        if len(completed_batches) < len(batch_ids):
            print("  Waiting for 1 minute before checking again...")
            time.sleep(60)
    print(f"\n✓ All batches finished!")

    # Step 3: Download and merge results
    print(f"\nStep 3: Downloading and merging results...")
    scored_result = []
    
    for batch_id in batch_ids:
        batch_obj = openai_client.batches.retrieve(batch_id)
        if batch_obj.status != "completed":
            print(f"Skipping batch {batch_id_to_number[batch_id]} (status: {batch_obj.status})")
            continue

        if batch_obj.error_file_id:
            err = openai_client.files.content(batch_obj.error_file_id).text
            print(f"Batch {batch_id_to_number[batch_id]} errors:\n{err}")

        if not batch_obj.output_file_id:
            print(f"Batch {batch_id_to_number[batch_id]} has no output_file_id.")
            continue

        content = openai_client.files.content(batch_obj.output_file_id).text
        if not content.strip():
            print(f"Batch {batch_id_to_number[batch_id]} output is empty.")
            continue

        # Save batch results locally
        results_file = f"{model_name}_scoring_batch_{batch_id_to_number[batch_id]}_results.jsonl"
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(content)

        # Parse each line in results
        lines = content.splitlines()
        for line in lines:
            r = json.loads(line)
            custom_id = r["custom_id"]
            index = int(custom_id.split('-')[-1])
            parsed = r["response"]["body"]["choices"][0]["message"]["content"]
            if not parsed:
                continue
            parsed_json = json.loads(parsed)

            scored_result.append({
                "question": results[index]["question"],
                "student_answer": results[index]["answer"],
                "ground_truth_answers": results[index]["ground_truth"],
                "explanation": parsed_json["explanation"],
                "score": parsed_json["score"]
            })
        
        print(f"  Processed batch {batch_id_to_number[batch_id]} ({len(lines)} scores)")

    print(f"Collected {len(scored_result)} scored results from {model_name}.")

    # Save scored results to file
    output_filename = f"{model_name}-scored-{CURRENT_DATE}-hw2.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(scored_result, f, indent=2, ensure_ascii=False)

    # Calculate and display accuracy
    correct = sum(1 for r in scored_result if r['score'])
    accuracy = (correct / len(scored_result)) * 100 if scored_result else 0
    
    print(f"Scoring complete!")
    print(f"Saved scored results to {output_filename}")
    print(f"Accuracy: {correct}/{len(scored_result)} ({accuracy:.2f}%)")
    print(f"Completion rate: {completed}/{total} ({completion_rate:.2f}%)")


# ============================================================================
# QUESTION 5: Image Creation with GPT-image-1
# ============================================================================
def image_creation(quality, size):
    # Oh, when Dr. Mills did his capstone project ECE 492...
    prompt = "An AI-driven EHS platform that analyzes historical and real-time jobsite data to predict safety trends and prevent incidents."

    response = openai_client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        n=1,
        size=size,
        quality=quality
    )

    image_b64 = response.data[0].b64_json
    image_data = base64.b64decode(image_b64)

    filename = f"image_{quality}_{size}.png"
    with open(filename, "wb") as f:
        f.write(image_data)

    print(f"Saved {filename}")

# Simple menu to select options
while True:
    choice = input(
        "Enter 1-5:\n"
        "1. Minimal Reasoning (GPT-5-nano batch processing)\n"
        "2. Sequential Answering (Qwen3-8b)\n"
        "3. Scoring Both Model (GPT-5-nano and Qwen3-8b)\n"
        "4. Image Creation (GPT-image-1)\n"
        "5. Quit\n"
        "Choice: "
    )
    if choice == "1":
        gpt_5(questions)
    elif choice == "2":
        qwen3_8b(questions)
    elif choice == "3":

        gpt_filename = f"gpt-5-nano-{CURRENT_DATE}-hw2.json"
        qwen_filename = f"qwen3-8b-{CURRENT_DATE}-hw2.json"
        if(not os.path.exists(gpt_filename) or not os.path.exists(qwen_filename)):
            print("Please run options 1 and 2 first to generate the answer files.")
            continue
        scoring_result("gpt-5-nano", gpt_filename)
        scoring_result("qwen3-8b", qwen_filename)
    elif choice == "4":
        image_creation("medium", "1024x1024")
        image_creation("low", "1024x1024")
    elif choice == "5":
        print("Exiting program.")
        exit()
    else:
        print("Invalid choice. Please enter a number between 1 and 4.")