"""
CSC 4700 Homework 3: RAG and LLM Evaluation

Author: Daniel Guo
Instructor: Dr. Keith Mills

This homework builds on Homework 2. The program evaluates
multiple API-based LLMs on a subset of SQuAD 2.0, this time using
retrieval-augmented generation (RAG) to provide context to the models. It
then uses GPT-5-mini to score the correctness of each model's responses.

Allowed libraries:
- openai: For accessing GPT-5-nano and GPT-5-mini via OpenAI API
- dotenv: For loading environment variables from .env file
- pydantic: For structured data validation in scoring responses
- tiktoken: For token counting (imported but not actively used)
- chromadb: For storing and retrieving context chunks in RAG
- os: For file system operations and environment variable access
- json: For reading/writing JSON data files
- datetime: For timestamping output files
- time: For implementing delays in batch status polling
- base64: For decoding base64-encoded images

References:
Google and ChatGPT were used to clarify syntax, standard library usage,
and API interactions.
"""

from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb.utils import embedding_functions
import chromadb
import os
import json
import time
import base64

# Store current date globally to maintain consistency across file operations
CURRENT_DATE = datetime.now().strftime("%Y%m%d")

# Load the SQuAD 2.0 development dataset from JSON file
with open("dev-v2.0.json", "r", encoding="utf-8") as f:
    squad_data = json.load(f)

# Initialize Chroma collection with SQuAD contexts
chroma_collection = None

# ===========================================================================
# STEP 1: EXTRACT CONTEXT CHUNKS AND STORE IN CHROMADB
# ===========================================================================

# Load API keys from environment file (needed for embeddings)
load_dotenv('.env')

def extract_and_store_chunks(squad_data):
    """
    Extract all context passages from SQuAD and store them in a Chroma database.
    
    This function:
    - Loops through all paragraphs in the SQuAD JSON
    - Extracts the 'context' field from each paragraph
    - Embeds each context using text-embedding-3-small
    - Stores embeddings and metadata in a persistent Chroma collection
    
    Args:
        squad_data (dict): Loaded SQuAD 2.0 JSON data
    
    Returns:
        chromadb.Collection: The populated ChromaDB collection
    """
    # Initialize OpenAI embedding function
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    
    # Create persistent Chroma client
    client = chromadb.PersistentClient(path="./kb")
    
    # Get or create collection for SQuAD contexts
    col = client.get_or_create_collection(
        name="handbook",
        embedding_function=ef
    )
    
    print("Extracting contexts from SQuAD data...")
    contexts = []
    ids = []
    context_id = 0
    
    # Extract all unique contexts from nested SQuAD structure
    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            context_text = paragraph.get('context', '')
            if context_text.strip():
                contexts.append(context_text)
                ids.append(f"context_{context_id}")
                context_id += 1
    
    print(f"Found {len(contexts)} unique contexts")
    
    # Upsert contexts into Chroma
    if contexts:
        col.upsert(
            ids=ids,
            documents=contexts,
            metadatas=[{"source": "squad_v2.0"} for _ in contexts]
        )
        print(f"Stored {len(contexts)} contexts in ChromaDB collection")
    
    return col

# ===========================================================================
# STEP 2: SETUP AND DATA LOADING
# ===========================================================================

# List to store extracted questions with their ground truth answers
questions = []

# Extract first 500 answerable questions from nested SQuAD structure
# SQuAD structure: data -> paragraphs -> qas (question-answer pairs)
for article in squad_data['data']:
    for paragraph in article['paragraphs']:
        for qa in paragraph['qas']:
            # Skip impossible questions (those without answers)
            if not qa.get('is_impossible', False):
                # Extract unique answer texts from multiple annotators
                answer_texts = [ans['text'] for ans in qa.get('answers', [])]
                # Only add if there are actual answers provided
                if answer_texts:
                    questions.append({
                        "question": qa['question'],
                        "answers": answer_texts,
                        "id": qa.get('id', f'q_{len(questions)}')
                    })
            # Stop once we've collected 500 questions
            if len(questions) >= 500:
                break
        if len(questions) >= 500:
            break
    if len(questions) >= 500:
        break

print(f"Collected {len(questions)} questions.")

# Initialize ChromaDB with SQuAD contexts
print("\nInitializing ChromaDB with SQuAD contexts...")
chroma_collection = extract_and_store_chunks(squad_data)

# ===========================================================================
# SETUP OPENAI AND OPENROUTER CLIENTS
# ===========================================================================

# Initialize OpenAI client for GPT-5-nano and GPT-5-mini models
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize OpenRouter client for Qwen3-8b model
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# ===========================================================================
# STEP 3: GPT-5-NANO USING OPENAI BATCH API WITH RAG
# ===========================================================================
# Global list to store GPT-5-nano answers
gpt_5_nano_answer = []
def gpt_5(questions, batch_size=100):
    """
    Process questions using OpenAI's Batch API with GPT-5-nano model and RAG.
    
    This function divides questions into batches, retrieves relevant context
    from ChromaDB using semantic search, and submits RAG-augmented requests
    to OpenAI's Batch API.
    
    Args:
        questions (list): List of question dictionaries with 'question',
                         'answers', and 'id' keys
        batch_size (int): Number of questions per batch (default: 100)
    
    Returns:
        None: Results are saved to global gpt_5_nano_answer list and
              written to JSON file
    """
    global gpt_5_nano_answer
    # Reset global answer list for fresh run
    gpt_5_nano_answer = []
    
    print("Starting GPT-5-nano batch processing...")
    
    # List to track submitted batch IDs
    batch_ids = []
    batch_number = 0
    # Mapping from batch ID to human-readable batch number
    batch_id_to_number = {}
    
    # Step 1: Create and submit batches
    for i in range(0, len(questions), batch_size):
        # Extract current batch of questions
        batch_questions = questions[i:i+batch_size]
        batch_number += 1
        
        # Create JSONL file with batch requests
        input_file = f"batch_{batch_number}_requests.jsonl"
        with open(input_file, 'w', encoding='utf-8') as f:
            for j, q in enumerate(batch_questions):
                # RAG: Query Chroma to retrieve 5 most relevant contexts
                rag_results = chroma_collection.query(
                    query_texts=[q["question"]],
                    n_results=5
                )
                
                # Extract context documents from query results
                contexts = rag_results.get("documents", [[]])[0]
                context_str = "\n\n".join(contexts) if contexts else ""
                
                # Format user message with context
                context_prompt = (
                    f"Context:\n{context_str}\n\n"
                    f"Question: {q['question']}"
                )
                
                # Each request includes custom_id for tracking
                request = {
                    "custom_id": f"batch{batch_number}-q{j}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-5-nano",
                        # System prompt instructs brief, concise answers
                        "messages": [
                            {
                                "role": "system",
                                "content":  "Answer concise from context, "
                                            "1-5 words. Do not explain "
                                            "your reasoning. If ambiguous, provide the "
                                            "most common answer."
                            },
                            {"role": "user", "content": context_prompt}
                        ],
                        # Explicitly request text response format
                        "response_format": {"type": "text"},
                    }
                }
                # Write each request as separate JSON line
                f.write(json.dumps(request) + '\n')

        # Upload JSONL file to OpenAI for batch processing
        batch_input_file = openai_client.files.create(
            file=open(input_file, "rb"),
            purpose="batch"
        )
        
        # Create batch job with 24-hour completion window
        batch = openai_client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        # Track batch ID and mapping
        batch_ids.append(batch.id)
        batch_id_to_number[batch.id] = batch_number
        print(f"  Batch {batch_number}: {batch.id} submitted "
              f"({len(batch_questions)} questions)")
    
    print(f"\n✓ All {len(batch_ids)} batch(es) submitted!")

    # Step 2: Wait for all batches to complete
    print(f"\nStep 2: Waiting for batches to complete...")
    # List to track completed batch IDs
    completed_batches = []
    
    # Poll until all batches are done
    while len(completed_batches) < len(batch_ids):
        for batch_id in batch_ids:
            # Skip already completed batches
            if batch_id in completed_batches:
                continue
            
            # Check current status of batch
            batch_status = openai_client.batches.retrieve(batch_id)
            if batch_status.status == "completed":
                completed_batches.append(batch_id)
                print(f"  Batch {batch_id_to_number[batch_id]} completed.")
        
        # Wait before next status check if not all done
        if len(completed_batches) < len(batch_ids):
            print("  Waiting for 1 minute before checking again...")
            time.sleep(60)
    
    print(f"\n✓ All batches finished!")

    # Step 3: Download and merge results
    print(f"\nStep 3: Downloading and merging results...")
    
    for batch_id in completed_batches:
        # Retrieve batch metadata
        batch_obj = openai_client.batches.retrieve(batch_id)
        
        # Skip if batch didn't complete successfully
        if batch_obj.status != "completed":
            continue

        # Check for and display any errors
        if batch_obj.error_file_id:
            error_content = openai_client.files.content(
                batch_obj.error_file_id
            ).text
            print(f"Batch {batch_id_to_number[batch_id]} errors:\n"
                  f"{error_content}")

        # Ensure output file exists
        if not batch_obj.output_file_id:
            print(f"Batch {batch_id_to_number[batch_id]} has no "
                  f"output_file_id.")
            continue

        # Download results as JSONL text
        result_content = openai_client.files.content(
            batch_obj.output_file_id
        ).text
        
        # Skip empty outputs
        if not result_content.strip():
            print(f"Batch {batch_id_to_number[batch_id]} output is empty.")
            continue

        # Save batch results locally for debugging
        results_file = f"batch_{batch_id_to_number[batch_id]}_results.jsonl"
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(result_content)

        # Parse each line in JSONL results
        for line in result_content.splitlines():
            result = json.loads(line)
            
            # Skip malformed results
            if "response" not in result:
                continue
            
            # Extract batch number and question index from custom_id
            custom_id = result['custom_id']
            batch_num, q_num = map(
                int,
                custom_id.replace('batch', '').split('-q')
            )
            
            # Calculate original question index across all batches
            original_question = questions[(batch_num - 1) * batch_size + q_num]
            
            # Extract answer from nested response structure
            choices = result['response']['body'].get('choices', [])
            if not choices:
                continue
            
            answer = choices[0]['message']['content']
            
            # Append to global results list
            gpt_5_nano_answer.append({
                "question": original_question['question'],
                "answer": answer,
                "ground_truth": original_question['answers']
            })
    
    print(f"Collected {len(gpt_5_nano_answer)} answers from GPT-5-nano.")

    # Save consolidated results to timestamped JSON file
    filename = f"gpt-5-nano-{CURRENT_DATE}-hw3.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(gpt_5_nano_answer, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(gpt_5_nano_answer)} GPT-5-nano answers to {filename}")


# ===========================================================================
# STEP 4: QWEN3-8B USING OPENROUTER API WITH RAG
# ===========================================================================

# Global list to store Qwen3-8b answers
qwen3_8b_answer = []


def qwen3_8b(questions):
    """
    Process questions sequentially using Qwen3-8b model via OpenRouter API with RAG.
    
    This function sends questions one at a time to the Qwen3-8b model,
    retrieving relevant context from ChromaDB and including it with each query.
    
    Args:
        questions (list): List of question dictionaries with 'question',
                         'answers', and 'id' keys
    
    Returns:
        None: Results are saved to global qwen3_8b_answer list and
              written to JSON file
    """
    global qwen3_8b_answer
    # Reset global answer list for fresh run
    qwen3_8b_answer = []
    
    print("Starting Qwen3-8b sequential processing...")
    print(f"Processing {len(questions)} questions with RAG context retrieval...")
    
    # Process each question individually
    for i, q in enumerate(questions):
        # Show progress for each question
        print(f"Processing question {i + 1}/{len(questions)}...", end='\r')
        
        # RAG: Query Chroma to retrieve 5 most relevant contexts
        rag_results = chroma_collection.query(
            query_texts=[q["question"]],
            n_results=5
        )
        
        # Extract context documents from query results
        contexts = rag_results.get("documents", [[]])[0]
        context_str = "\n\n".join(contexts) if contexts else ""
        
        # Format user message with context
        context_prompt = (
            f"Context:\n{context_str}\n\n"
            f"Question: {q['question']}"
        )
        
        # Make API call to Qwen3-8b model (retry on non-200)
        while True:
            try:
                response = openrouter_client.chat.completions.create(
                    model='qwen/qwen3-8b',
                    messages=[
                        {
                            "role": "system",
                            "content": "Answer concise from context, "
                                      "1-5 words. Do not explain "
                                      "your reasoning. If ambiguous, provide the "
                                      "most common answer."
                        },
                        {"role": "user", "content": context_prompt}
                    ],
                    # Lower temperature for more deterministic answers
                    temperature=0.3,
                    top_p=1,
                    n=1
                )
                status = getattr(response, "status_code", 200)
                if status != 200:
                    print(f"\nNon-200 status {status} on question {i + 1}. Retrying...")
                    time.sleep(2)
                    continue
                break
            except Exception as e:
                status = getattr(e, "status_code", None)
                if status is not None and status != 200:
                    print(f"\nNon-200 status {status} on question {i + 1}. Retrying...")
                else:
                    print(f"\nRequest error on question {i + 1}: {str(e)[:80]}")
                time.sleep(2)

        # Append question-answer pair with ground truth
        qwen3_8b_answer.append({
            "question": q['question'],
            # Handle potential None response
            "answer": response.choices[0].message.content or "",
            "ground_truth": q['answers']
        })
        
        # Display progress milestone every 10 questions
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(questions)} questions" + " " * 20)
            
    # Save results to timestamped JSON file
    filename = f"qwen3-8b-{CURRENT_DATE}-hw3.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(qwen3_8b_answer, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(qwen3_8b_answer)} Qwen3-8b answers to {filename}")


# ===========================================================================
# STEP 5: SCORING THE MODEL ANSWERS
# ===========================================================================

class ScoringResponse(BaseModel):
    """
    Pydantic model for structured scoring response validation.
    
    Attributes:
        explanation (str): Short explanation of scoring decision
        score (bool): True if answer is correct, False otherwise
    """
    explanation: str = Field(
        description="A short explanation of why the student's answer "
                   "was correct or incorrect"
    )
    score: bool = Field(
        description="true if the student's answer was correct, "
                   "false if it was incorrect"
    )


def scoring_result(model_name, result_file, batch_size=100):
    """
    Score model responses using GPT-5-mini via OpenAI Batch API.
    
    This function loads model responses, creates batches of scoring requests,
    submits them to GPT-5-mini for evaluation, and calculates accuracy.
    
    Args:
        model_name (str): Name of model being scored (for file naming)
        result_file (str): Path to JSON file containing model responses
        batch_size (int): Number of scoring requests per batch (default: 100)
    
    Returns:
        None: Results are saved to JSON file and statistics printed
    """
    print(f"Starting scoring for {model_name}...")
    
    # Load model responses to be scored
    with open(result_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Calculate completion rate (non-empty answers)
    total = len(results)
    completed = sum(1 for r in results if str(r.get("answer", "")).strip())
    completion_rate = (completed / total) * 100 if total else 0

    # Step 1: Create and submit batches
    print(f"\nStep 1: Creating batches of {batch_size} questions...")
    
    # List to track submitted batch IDs
    batch_ids = []
    batch_number = 0
    # Mapping from batch ID to human-readable batch number
    batch_id_to_number = {}
    
    # Create batches of scoring requests
    for i in range(0, len(results), batch_size):
        batch_results = results[i:i+batch_size]
        batch_number += 1
        
        # Create JSONL file with scoring requests
        input_file = (f"{model_name}_scoring_batch_{batch_number}_"
                     f"requests.jsonl")
        
        with open(input_file, "w", encoding="utf-8") as f:
            for j, result in enumerate(batch_results):
                # Calculate global index across all batches
                global_index = i + j
                
                # Each request includes custom_id for result matching
                request = {
                    "custom_id": (f"{model_name}-batch{batch_number}-"
                                f"score-{global_index}"),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-5-mini",
                        "messages": [
                            {
                                "role": "system",
                                "content":
                                "You are a teacher tasked with determining "
                                "whether a student's answer to a question "
                                "was correct, based on a set of possible "
                                "correct answers. You must only use the "
                                "provided possible correct answers to "
                                "determine if the student's response was "
                                "correct."
                            },
                            {
                                "role": "user",
                                "content":
                                f"Question: {result['question']}\n"
                                f"Student's Response: {result['answer']}\n"
                                f"Possible Correct Answers:\n"
                                f"{result['ground_truth']}\n"
                                f"Your response should only be a valid "
                                f"Json as shown below:\n"
                                "{\n"
                                '  "explanation" (str): A short '
                                'explanation of why the student\'s answer '
                                'was correct or incorrect.,\n'
                                '  "score" (bool): true if the student\'s '
                                'answer was correct, false if it was '
                                'incorrect\n'
                                "}\n"
                                "Your response:"
                            }
                        ],
                        # Use JSON schema to enforce structured output
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
                # Write each request as separate JSON line
                f.write(json.dumps(request) + "\n")

        # Upload JSONL file to OpenAI for batch processing
        batch_input_file = openai_client.files.create(
            file=open(input_file, "rb"),
            purpose="batch"
        )

        # Create batch job with 24-hour completion window
        batch = openai_client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        # Track batch ID and mapping
        batch_ids.append(batch.id)
        batch_id_to_number[batch.id] = batch_number
        print(f"  Batch {batch_number}: {batch.id} submitted "
              f"({len(batch_results)} questions)")

    print(f"\n✓ All {len(batch_ids)} batch(es) submitted!")

    # Step 2: Wait for all batches to complete
    print(f"\nStep 2: Waiting for batches to complete...")
    # List to track completed batch IDs
    completed_batches = []
    
    # Poll until all batches are done
    while len(completed_batches) < len(batch_ids):
        for batch_id in batch_ids:
            # Skip already completed batches
            if batch_id in completed_batches:
                continue
            
            # Check current status of batch
            batch_status = openai_client.batches.retrieve(batch_id)
            
            if batch_status.status == "completed":
                completed_batches.append(batch_id)
                print(f"  Batch {batch_id_to_number[batch_id]} completed.")
            elif batch_status.status in ("failed", "cancelled", "expired"):
                # Handle failed batches gracefully
                print(f"  Batch {batch_id_to_number[batch_id]} ended with "
                      f"status: {batch_status.status}")
                
                # Display error details if available
                if batch_status.error_file_id:
                    err = openai_client.files.content(
                        batch_status.error_file_id
                    ).text
                    print(f"  Batch {batch_id_to_number[batch_id]} errors:\n"
                          f"{err}")
                
                # Mark as done to avoid infinite loop
                completed_batches.append(batch_id)
        
        # Wait before next status check if not all done
        if len(completed_batches) < len(batch_ids):
            print("  Waiting for 1 minute before checking again...")
            time.sleep(60)
    
    print(f"\n✓ All batches finished!")

    # Step 3: Download and merge results
    print(f"\nStep 3: Downloading and merging results...")
    # List to accumulate all scored results
    scored_result = []
    
    for batch_id in batch_ids:
        # Retrieve batch metadata
        batch_obj = openai_client.batches.retrieve(batch_id)
        
        # Skip if batch didn't complete successfully
        if batch_obj.status != "completed":
            print(f"Skipping batch {batch_id_to_number[batch_id]} "
                  f"(status: {batch_obj.status})")
            continue

        # Check for and display any errors
        if batch_obj.error_file_id:
            err = openai_client.files.content(
                batch_obj.error_file_id
            ).text
            print(f"Batch {batch_id_to_number[batch_id]} errors:\n{err}")

        # Ensure output file exists
        if not batch_obj.output_file_id:
            print(f"Batch {batch_id_to_number[batch_id]} has no "
                  f"output_file_id.")
            continue

        # Download results as JSONL text
        content = openai_client.files.content(batch_obj.output_file_id).text
        
        # Skip empty outputs
        if not content.strip():
            print(f"Batch {batch_id_to_number[batch_id]} output is empty.")
            continue

        # Save batch results locally for debugging
        results_file = (f"{model_name}_scoring_batch_"
                       f"{batch_id_to_number[batch_id]}_results.jsonl")
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(content)

        # Parse each line in JSONL results
        lines = content.splitlines()
        for line in lines:
            r = json.loads(line)
            
            # Extract original index from custom_id
            custom_id = r["custom_id"]
            index = int(custom_id.split('-')[-1])
            
            # Extract JSON response content
            parsed = r["response"]["body"]["choices"][0]["message"]["content"]
            
            # Skip empty responses
            if not parsed:
                continue
            
            # Parse JSON string to dictionary
            parsed_json = json.loads(parsed)

            # Append complete scored result with all metadata
            scored_result.append({
                "question": results[index]["question"],
                "student_answer": results[index]["answer"],
                "ground_truth_answers": results[index]["ground_truth"],
                "explanation": parsed_json["explanation"],
                "score": parsed_json["score"]
            })
        
        print(f"  Processed batch {batch_id_to_number[batch_id]} "
              f"({len(lines)} scores)")

    print(f"Collected {len(scored_result)} scored results from "
          f"{model_name}.")

    # Save consolidated scored results to timestamped JSON file
    output_filename = f"{model_name}-RAG-{CURRENT_DATE}-hw3.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(scored_result, f, indent=2, ensure_ascii=False)

    # Calculate and display accuracy metrics
    correct = sum(1 for r in scored_result if r['score'])
    accuracy = (correct / len(scored_result)) * 100 if scored_result else 0
    
    print(f"Scoring complete!")
    print(f"Saved scored results to {output_filename}")
    print(f"Accuracy: {correct}/{len(scored_result)} ({accuracy:.2f}%)")
    print(f"Completion rate: {completed}/{total} ({completion_rate:.2f}%)")


# ===========================================================================
# STEP 6: IMAGE CREATION WITH GPT-IMAGE-1
# ===========================================================================

def image_creation(quality, size):
    """
    Generate AI images using OpenAI's gpt-image-1 model.
    
    Creates an image based on the EHS platform prompt and saves it as
    a PNG file with quality and size in the filename.
    
    Args:
        quality (str): Image quality setting ('low', 'medium', or 'high')
        size (str): Image dimensions (e.g., '1024x1024', '512x512')
    
    Returns:
        None: Image is saved to file with descriptive name
    """
    # Prompt describing Dr. Mills' capstone project
    prompt = ("An AI-driven EHS platform that analyzes historical and "
              "real-time jobsite data to predict safety trends and "
              "prevent incidents.")

    # Generate image using OpenAI's image API
    response = openai_client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        n=1,  # Generate one image
        size=size,
        quality=quality
    )

    # Extract base64-encoded image data
    image_b64 = response.data[0].b64_json
    # Decode base64 to binary image data
    image_data = base64.b64decode(image_b64)

    # Save image with descriptive filename
    filename = f"image_{quality}_{size}.png"
    with open(filename, "wb") as f:
        f.write(image_data)

    print(f"Saved {filename}")


# ===========================================================================
# MAIN MENU INTERFACE
# ===========================================================================

# Interactive menu loop for running different homework tasks
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
        # Run GPT-5-nano batch processing on all 500 questions
        gpt_5(questions)
        
    elif choice == "2":
        # Run Qwen3-8b sequential processing on all 500 questions
        qwen3_8b(questions)
        
    elif choice == "3":
        # Score both models' responses using GPT-5-mini
        
        # Construct expected filenames based on current date
        gpt_filename = f"gpt-5-nano-{CURRENT_DATE}-hw3.json"
        qwen_filename = f"qwen3-8b-{CURRENT_DATE}-hw3.json"
        
        # Check if answer files exist before attempting to score
        if (not os.path.exists(gpt_filename) or
                not os.path.exists(qwen_filename)):
            print("Please run options 1 and 2 first to generate the "
                  "answer files.")
            continue
        
        # Score both models' responses
        scoring_result("gpt-5-nano", gpt_filename)
        scoring_result("qwen3-8b", qwen_filename)
        
    elif choice == "4":
        # Generate medium and low quality images
        image_creation("medium", "1024x1024")
        image_creation("low", "1024x1024")
        
    elif choice == "5":
        # Exit program
        print("Exiting program.")
        exit()
        
    else:
        # Handle invalid menu choices
        print("Invalid choice. Please enter a number between 1 and 5.")