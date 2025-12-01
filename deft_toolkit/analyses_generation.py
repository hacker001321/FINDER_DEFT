#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO PersonalAI team. All rights reserved.
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

# Parts of this code are adapted from TopicGPT
# https://github.com/ArikReuter/TopicGPT
# Licensed under MIT License


from deft_toolkit.utils import APIClient
import argparse
import traceback
import random
import re
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from urllib.parse import urldefrag
import pandas as pd


def evaluate_records(
    api_client: APIClient,
    data_list: List[Dict[str, Any]],
    prompt_template: Dict[str, str],
    temperature: float,
    max_workers: int
) -> List[Dict[str, Any]]:
    """
    Perform failure analysis on execution records using an LLM.

    Parameters:
    - api_client (APIClient): The API client for LLM service.
    - data_list (List[Dict[str, Any]]): A list of input data records, each containing 'id', 'source', 'question', and 'article'.
    - prompt_template (Dict[str, str]): A dictionary with 'zh' and 'en' keys containing prompt templates with %s placeholders.
    - temperature (float): The sampling temperature for LLM generation.
    - max_workers (int): The maximum number of concurrent threads for processing.

    Returns:
    - List[Dict[str, Any]]: The updated list of data records with 'failure_analysis' field added or preserved.
    """

    # Separate items that need processing from those already analyzed
    to_process = []
    skipped = []
    for data in data_list:
        if "failure_analysis" not in data or not str(data["failure_analysis"]).strip():
            to_process.append(data)
        else:
            skipped.append(data)

    print(f"Total items: {len(data_list)}, To process: {len(to_process)}, Skipped: {len(skipped)}")

    random.shuffle(to_process)

    def process_item(data: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """
        Process a single execution record to generate failure analysis.

        Parameters:
        - data (Dict[str, Any]): A single data record containing 'question' and 'article' fields.

        Returns:
        - Tuple[Dict[str, Any], str]: A tuple containing the original data record and the generated analysis response.
        """
        try:
            question = data["question"]
            article = data["article"]

            # Remove URL fragments to avoid noise
            article = re.sub(r'https?://[^\s\)]+', lambda m: urldefrag(m.group(0))[0], article)

            # Choose prompt based on language detection
            if any('\u4e00' <= ch <= '\u9fff' for ch in question):
                prompt = prompt_template["zh"] % (question, article)
            else:
                prompt = prompt_template["en"] % (question, article)

            response = api_client.iterative_prompt(prompt, temperature)
            return data, response
        except Exception:
            traceback.print_exc()
            return data, "error"

    results = []
    if to_process:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_item, data): data for data in to_process}
            with tqdm(total=len(to_process), desc="Generating Failure Analysis") as pbar:
                for future in as_completed(futures):
                    data, response = future.result()
                    results.append((data, response))
                    pbar.update(1)

    # Reconstruct the full data list with results
    new_data_list = []
    result_dict = {r[0]["id"]: r[1] for r in results}

    for data in data_list:
        if "failure_analysis" not in data or not str(data["failure_analysis"]).strip():
            failure_analysis = result_dict[data["id"]]
            new_data = {
                "id": data["id"],
                "question": data["question"],
                "article": data["article"],
                "analysis_model": api_client.model,
                "failure_analysis": failure_analysis,
            }
            # Preserve optional fields if they exist
            if "source" in data:
                new_data["source"] = data["source"]
            new_data_list.append(new_data)
        else:
            new_data_list.append(data)

    return new_data_list


def generate_analyses(
    model: str,
    data: str,
    out_file: str,
    prompt_zh_file: str,
    prompt_en_file: str,
    max_workers: int,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
) -> None:
    """
    Run failure analysis on execution records using a specified LLM.

    Parameters:
    - model (str): The name of the LLM model to use.
    - data (str): The path to the input JSONL file, where each line contains 'id', 'source', 'question', and 'article'.
    - out_file (str): The path to the output JSONL file with added 'failure_analysis' field.
    - prompt_zh_file (str): The path to the Chinese prompt template file (uses %s formatting).
    - prompt_en_file (str): The path to the English prompt template file (uses %s formatting).
    - max_workers (int): The maximum number of parallel API requests.
    - llm_api_key (Optional[str]): The API key for LLM service. Uses global config if not provided.
    - llm_base_url (Optional[str]): The base URL for LLM API. Uses global config if not provided.
    """
    api_client = APIClient(model=model, llm_api_key=llm_api_key, llm_base_url=llm_base_url)
    temperature = 0.3

    print("-------------------")
    print("Initializing failure analysis...")
    print(f"Model: {model}")
    print(f"Data: {data}")
    print(f"Output file: {out_file}")
    print(f"Max workers: {max_workers}")
    print("-------------------")

    # Load data
    df = pd.read_json(data, lines=True)
    data_list = df.to_dict('records')

    # Load prompt templates
    try:
        with open(prompt_zh_file, "r", encoding="utf-8") as f:
            prompt_zh = f.read().strip()
        with open(prompt_en_file, "r", encoding="utf-8") as f:
            prompt_en = f.read().strip()
    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Failed to load prompt files: {e}")

    prompt_template = {
        "zh": prompt_zh,
        "en": prompt_en
    }

    # Run analysis
    new_data_list = evaluate_records(
        api_client=api_client,
        data_list=data_list,
        prompt_template=prompt_template,
        temperature=temperature,
        max_workers=max_workers
    )

    # Save results
    result_df = pd.DataFrame(new_data_list)
    json_str = result_df.to_json(lines=True, orient='records', force_ascii=False)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(json_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model to use")
    parser.add_argument(
        "--data",
        type=str,
        default="data/input/records.jsonl",
        help="Input JSONL file to run analysis generation on")
    parser.add_argument(
        "--out_file",
        type=str,
        default="data/input/records_with_analysis.jsonl",
        help="Output JSONL file to write results to")
    parser.add_argument(
        "--prompt_zh_file",
        type=str,
        default="prompt/analyses_generation_zh.txt",
        help="Chinese prompt template file"
    )
    parser.add_argument(
        "--prompt_en_file",
        type=str,
        default="prompt/analyses_generation_en.txt",
        help="English prompt template file"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Maximum number of parallel threads")

    args = parser.parse_args()

    generate_analyses(
        model=args.model,
        data=args.data,
        out_file=args.out_file,
        prompt_en_file=args.prompt_en_file,
        prompt_zh_file=args.prompt_zh_file,
        max_workers=args.max_workers,
    )
