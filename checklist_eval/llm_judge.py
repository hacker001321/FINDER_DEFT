# Copyright 2025 The OPPO Inc. PersonalAI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from openai import OpenAI, AsyncOpenAI
import time
import re
import os
from pathlib import Path
import asyncio
import argparse
from typing import List, Dict, Any

# ****************** Configuration ******************
# API credentials from environment variables
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") 
BASE_URL = os.getenv("BASE_URL")

# Default configuration (can be overridden by command line arguments)
DEFAULT_INPUT_FOLDER = "./data"  
DEFAULT_CHECKLIST_FILE = "./data/checklist.jsonl"
DEFAULT_OUTPUT_FOLDER = "./evaluation_checklist_results"  
DEFAULT_MAX_CONCURRENT_REQUESTS = 10
DEFAULT_REQUEST_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 5

# Global variables (will be set in main)
INPUT_FOLDER = DEFAULT_INPUT_FOLDER
CHECKLIST_FILE = DEFAULT_CHECKLIST_FILE
OUTPUT_FOLDER = DEFAULT_OUTPUT_FOLDER
MAX_CONCURRENT_REQUESTS = DEFAULT_MAX_CONCURRENT_REQUESTS
REQUEST_TIMEOUT = DEFAULT_REQUEST_TIMEOUT
MAX_RETRIES = DEFAULT_MAX_RETRIES
# ***************************************************

# Initialize API clients
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
async_client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

async def evaluate_checklist_item_async(article_text: str, checklist_description: str, 
                                        article_id: str, item_title: str, semaphore: asyncio.Semaphore) -> bool:
    """
    Asynchronously evaluate a single checklist item with timeout and retry logic.
    
    Args:
        article_text: The text content to evaluate
        checklist_description: The checklist criterion description
        article_id: Unique identifier for the article
        item_title: Title of the checklist item
        semaphore: Asyncio semaphore for concurrency control
        
    Returns:
        bool: True if the criterion is met, False otherwise
    """
    async with semaphore:  # Control concurrency
        # Use string concatenation to avoid formatting issues
        prompt_content = (
            "Please evaluate whether the following text meets the checklist criteria:\n\n"
            "[Checklist Criteria]\n" + str(checklist_description) + "\n\n"
            "[Text to Evaluate]\n" + str(article_text) + "\n\n"
            "[Requirements] Carefully read the text and determine if it fully meets the above criteria.\n"
            "[Output Requirements]\n"
            "1. Carefully analyze the text and judge whether it fully meets the above criteria.\n"
            "2. Only answer: Yes or No (no explanation needed)\n"
            "3. Do not output any other content.\n"
            "4. Directly output: Yes or No"
        )
        
        # Retry logic with exponential backoff
        for retry in range(MAX_RETRIES):
            try:
                # Add timeout control
                response = await asyncio.wait_for(
                    async_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a professional text evaluation expert. After carefully analyzing the text, only answer 'Yes' or 'No'."
                            },
                            {
                                "role": "user", 
                                "content": prompt_content
                            }
                        ],
                        temperature=0.1,
                        max_tokens=2000
                    ),
                    timeout=REQUEST_TIMEOUT
                )
                
                if response.choices:
                    msg = response.choices[0].message
                    answer_text = msg.content
                    
                    # If content is empty, try to extract answer from reasoning_content
                    if not answer_text:
                        reasoning = getattr(msg, 'reasoning_content', None)
                        if reasoning:
                            last_lines = reasoning.strip().split('\n')[-5:]
                            answer_text = '\n'.join(last_lines)
                    
                    if answer_text:
                        answer = answer_text.strip().lower()
                        is_positive = 'yes' in answer or 'true' in answer
                        is_negative = 'no' in answer or 'false' in answer
                        
                        if is_positive and not is_negative:
                            print(f"  [{article_id}] {item_title}: Yes ✓")
                            return True
                        elif is_negative:
                            print(f"  [{article_id}] {item_title}: No ✗")
                            return False
                        else:
                            print(f"  [{article_id}] {item_title}: Unable to determine, default No")
                            return False
                    else:
                        print(f"  [{article_id}] {item_title}: Empty response")
                        return False
                else:
                    print(f"  [{article_id}] {item_title}: No response")
                    return False
                    
            except asyncio.TimeoutError:
                print(f"  [{article_id}] {item_title}: Timeout (attempt {retry + 1}/{MAX_RETRIES})")
                if retry < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** retry)  # Exponential backoff
                    continue
                else:
                    print(f"  [{article_id}] {item_title}: Timeout failed, skipping")
                    return False
                    
            except Exception as e:
                print(f"  [{article_id}] {item_title}: Error (attempt {retry + 1}/{MAX_RETRIES}) - {e}")
                if retry < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** retry)  # Exponential backoff
                    continue
                else:
                    print(f"  [{article_id}] {item_title}: Evaluation failed")
                    return False
        
        # Should not reach here
        return False

async def process_article_async(article_data: Dict, standard: Dict, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """
    Asynchronously process a single article against checklist criteria.
    
    Args:
        article_data: Dictionary containing article data with 'id' and 'article'/'prediction'
        standard: Dictionary containing checklist criteria
        semaphore: Asyncio semaphore for concurrency control
        
    Returns:
        Dict containing evaluation results
    """
    article_id = article_data['id']
    try:
        article_text = article_data['article']
    except KeyError:
        article_text = article_data.get('prediction', '') 
    
    article_evaluation = {
        "id": article_id,
        "topic": standard.get('topic', ''),
        "checklist_evaluations": [],
        "model_used": MODEL_NAME
    }
    
    # Evaluate all checklist items
    checklist_items = standard.get('checklist', [])
    
    # Create evaluation tasks
    tasks = []
    for i, checklist_item in enumerate(checklist_items):
        task = evaluate_checklist_item_async(
            article_text,
            checklist_item.get('description', ''),
            article_id,
            checklist_item.get('title', f'item_{i+1}'),
            semaphore
        )
        tasks.append((checklist_item, task))
    
    # Execute all evaluations concurrently
    for checklist_item, task in tasks:
        is_met = await task
        evaluation_item = {
            "checklist_title": checklist_item.get('title', ''),
            "is_met": is_met,
            "model_used": MODEL_NAME
        }
        article_evaluation['checklist_evaluations'].append(evaluation_item)
    
    return article_evaluation

async def process_single_file_async(input_file_path: str, output_file_path: str, standards: List[Dict]):
    """
    Asynchronously process a single JSONL file with checklist evaluation.
    
    Args:
        input_file_path: Path to input JSONL file
        output_file_path: Path to output results file
        standards: List of checklist standards
        
    Returns:
        Tuple of (articles_evaluated, successful_checks, total_checks)
    """
    print(f"\n{'='*60}")
    print(f"Processing file: {os.path.basename(input_file_path)}")
    print(f"{'='*60}")
    
    # Read article data
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            articles = [json.loads(line) for line in f]
        print(f"Successfully loaded {len(articles)} articles")
    except Exception as e:
        print(f"Failed to read file: {e}")
        return 0, 0, 0

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Prepare evaluation tasks
    tasks = []
    for article_data in articles:
        article_id = article_data['id']
        try:
            article_text = article_data['article']
        except KeyError:
            article_text = article_data.get('prediction', '')
        
        if not article_text or article_text == "[生成失败]":
            print(f"Skipping article ID: {article_id}")
            continue
            
        standard = next((s for s in standards if s['id'] == article_id), None)
        if not standard:
            print(f"No evaluation standard found for article {article_id}")
            continue
        
        print(f"=== Preparing evaluation for article ID: {article_id} ===")
        task = process_article_async(article_data, standard, semaphore)
        tasks.append(task)
    
    # Execute all article evaluations concurrently
    print(f"\nStarting concurrent evaluation of {len(tasks)} articles (max concurrency: {MAX_CONCURRENT_REQUESTS})...")
    print(f"Timeout: {REQUEST_TIMEOUT}s, Max retries: {MAX_RETRIES}\n")
    
    batch_start_time = time.time()
    evaluation_results = await asyncio.gather(*tasks)
    batch_elapsed = time.time() - batch_start_time
    print(f"\n✓ Completed evaluation of {len(evaluation_results)} articles in {batch_elapsed:.2f}s")
    
    # Calculate statistics
    total_checks = 0
    successful_checks = 0
    for result in evaluation_results:
        for item in result['checklist_evaluations']:
            total_checks += 1
            if item['is_met']:
                successful_checks += 1

    # Save results
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for result in evaluation_results:
                f.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(f"\nEvaluation results saved to: {output_file_path}")
    except Exception as e:
        print(f"Failed to save: {e}")

    print(f"\nFile {os.path.basename(input_file_path)} processing completed:")
    print(f"Articles evaluated: {len(evaluation_results)}")
    print(f"Checks passed: {successful_checks}/{total_checks}")
    
    return len(evaluation_results), successful_checks, total_checks

async def main_async():
    """Main async function for batch evaluation."""
    print("Starting batch evaluation (async mode)...")
    print(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    
    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Output folder: {OUTPUT_FOLDER}")
    
    # Load evaluation standards
    try:
        with open(CHECKLIST_FILE, 'r', encoding='utf-8') as f:
            standards = [json.loads(line) for line in f]
        print(f"Successfully loaded {len(standards)} evaluation standards")
    except Exception as e:
        print(f"Failed to read checklist file: {e}")
        return

    # Get all .jsonl files in input folder
    input_folder = Path(INPUT_FOLDER)
    jsonl_files = list(input_folder.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No .jsonl files found in {INPUT_FOLDER}")
        return
    
    print(f"\nFound {len(jsonl_files)} .jsonl file(s) to process")
    for f in jsonl_files:
        print(f"  - {f.name}")
    
    # Initialize statistics
    total_files_processed = 0
    total_articles_evaluated = 0
    total_checks_sum = 0
    total_successful_checks = 0
    
    start_time = time.time()
    
    # Process each file
    for input_file in jsonl_files:
        output_file = Path(OUTPUT_FOLDER) / f"eval_{input_file.name}"
        
        # Check if output file already exists
        if output_file.exists():
            print(f"\n⚠️  Skipping {input_file.name} - evaluation results already exist: {output_file.name}")
            continue
        
        articles_count, successful, total = await process_single_file_async(
            str(input_file), 
            str(output_file), 
            standards
        )
        
        total_files_processed += 1
        total_articles_evaluated += articles_count
        total_checks_sum += total
        total_successful_checks += successful
    
    elapsed_time = time.time() - start_time
    
    # Print overall statistics
    print(f"\n{'='*60}")
    print(f"All files processed!")
    print(f"{'='*60}")
    print(f"Files processed: {total_files_processed}")
    print(f"Articles evaluated: {total_articles_evaluated}")
    print(f"Checks passed: {total_successful_checks}/{total_checks_sum}")
    if total_checks_sum > 0:
        print(f"Pass rate: {total_successful_checks/total_checks_sum*100:.2f}%")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Average per check: {elapsed_time/total_checks_sum:.2f}s" if total_checks_sum > 0 else "")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Checklist-based evaluation for deep research articles using LLM judge"
    )
    parser.add_argument(
        "--input_folder", 
        type=str, 
        default=DEFAULT_INPUT_FOLDER, 
        help=f"Input folder containing .jsonl files (default: {DEFAULT_INPUT_FOLDER})"
    )
    parser.add_argument(
        "--output_folder", 
        type=str, 
        default=DEFAULT_OUTPUT_FOLDER, 
        help=f"Output folder for evaluation results (default: {DEFAULT_OUTPUT_FOLDER})"
    )
    parser.add_argument(
        "--checklist_file", 
        type=str, 
        default=DEFAULT_CHECKLIST_FILE, 
        help=f"Checklist criteria file (default: {DEFAULT_CHECKLIST_FILE})"
    )
    parser.add_argument(
        "--max_concurrent_requests", 
        type=int, 
        default=DEFAULT_MAX_CONCURRENT_REQUESTS, 
        help=f"Maximum concurrent API requests (default: {DEFAULT_MAX_CONCURRENT_REQUESTS})"
    )
    parser.add_argument(
        "--request_timeout", 
        type=int, 
        default=DEFAULT_REQUEST_TIMEOUT, 
        help=f"API request timeout in seconds (default: {DEFAULT_REQUEST_TIMEOUT})"
    )
    parser.add_argument(
        "--max_retries", 
        type=int, 
        default=DEFAULT_MAX_RETRIES, 
        help=f"Maximum retry attempts (default: {DEFAULT_MAX_RETRIES})"
    )
    return parser.parse_args()

def main():
    """Synchronous entry point."""
    global INPUT_FOLDER, CHECKLIST_FILE, OUTPUT_FOLDER
    global MAX_CONCURRENT_REQUESTS, REQUEST_TIMEOUT, MAX_RETRIES
    
    # Parse command line arguments
    args = parse_args()
    
    # Update global configuration
    INPUT_FOLDER = args.input_folder
    CHECKLIST_FILE = args.checklist_file
    OUTPUT_FOLDER = args.output_folder
    MAX_CONCURRENT_REQUESTS = args.max_concurrent_requests
    REQUEST_TIMEOUT = args.request_timeout
    MAX_RETRIES = args.max_retries
    
    # Run async evaluation
    asyncio.run(main_async())

if __name__ == "__main__":
    main()