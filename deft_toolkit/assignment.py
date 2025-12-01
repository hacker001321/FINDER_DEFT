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


from deft_toolkit.utils import ModeTree, APIClient
import argparse
import traceback
import re
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd

def record_formatting(
        df: pd.DataFrame,
        lang: Optional[str] = None,
) -> pd.DataFrame:
    """
    Construct a unified 'record' field from 'question' and 'article' columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with 'question' and 'article' columns.
    - lang (Optional[str]): The language to force ('en' or 'zh'). If None, auto-detects from 'question'.

    Returns:
    - pd.DataFrame: A copy of the input DataFrame with an added 'record' column.
    """

    df = df.copy()
    record_list = []
    for _, row in df.iterrows():
        question = '' if pd.isna(row.get('question', '')) else str(row.get('question', '')).strip()
        article = '' if pd.isna(row.get('article', '')) else str(row.get('article', '')).strip()

        # Auto-detect language and format record based on language
        detected_lang = "zh" if re.search(r'[\u4e00-\u9fff]', question) else "en"
        use_lang = lang if lang is not None else detected_lang

        if use_lang == "zh":
            record = f"问题：\n{question}\n答案：\n{article}"
        else:
            record = f"Question: \n{question}\narticle: \n{article}\n"
        record_list.append(record)

    df['record'] = record_list
    return df

def annotate_records(
    api_client: APIClient,
    modes_root: ModeTree,
    records: List[str],
    assignment_prompt: str,
    temperature: float,
    max_workers: int
) -> List[str]:
    """
    Annotate a list of records with mode assignments using an LLM via parallel API calls.

    Parameters:
    - api_client (APIClient): The API client for LLM service.
    - modes_root (ModeTree): The root of the mode hierarchy tree.
    - records (List[str]): The list of record texts to classify.
    - assignment_prompt (str): The prompt template with placeholders {Record} and {Modes}.
    - temperature (float): The sampling temperature for LLM generation.
    - max_workers (int): The maximum number of concurrent threads.

    Returns:
    - List[str]: The list of responses aligned with input records order.
    """

    # Extract leaf nodes and format as descriptive strings
    leaf_info = modes_root.get_leaf_nodes_with_path()
    leaf_strings = [
        f"{node.name} (Level {node.lvl}): {node.desc}" if node.desc else f"{node.name} (Level {node.lvl})"
        for node, _ in leaf_info
    ]
    tree_str = "\n".join(leaf_strings)

    def process_record(idx: int, record: str) -> Tuple[int, str]:
        """Process a single record via LLM prompt."""
        try:
            prompt = assignment_prompt.format(Record=record, Modes=tree_str)
            response = api_client.iterative_prompt(prompt, temperature)
            return idx, response
        except Exception:
            traceback.print_exc()
            return idx, "Error"

    # Execute in parallel with progress tracking
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_record, i, record): i for i, record in enumerate(records)}
        with tqdm(total=len(records), desc="Processing records") as pbar:
            for future in as_completed(futures):
                idx, response = future.result()
                results.append((idx, response))
                pbar.update(1)

    # Reconstruct responses in original records order
    responses = [None] * len(records)
    for idx, resp in results:
        responses[idx] = resp

    return responses


def assign_modes(
    model: str,
    data: str,
    out_file: str,
    prompt_file: str,
    mode_file: str,
    max_workers: int,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
) -> None:
    """
    Assign modes to records using an LLM and save results to a JSONL file.

    Parameters:
    - model (str): The name of the LLM model to use (e.g., 'gpt-4o').
    - data (str): The path to the input JSONL file containing records.
    - out_file (str): The path to the output file for annotated results (JSONL).
    - prompt_file (str): The path to the prompt template file (must contain {Record} and {Modes} placeholders).
    - mode_file (str): The path to the mode hierarchy file (Markdown format).
    - max_workers (int): The maximum number of parallel API requests.
    - llm_api_key (Optional[str]): The API key for LLM service. Uses global config if not provided.
    - llm_base_url (Optional[str]): The base URL for LLM API. Uses global config if not provided.
    """

    api_client = APIClient(model=model, llm_api_key=llm_api_key, llm_base_url=llm_base_url)
    temperature = 0.1

    print("-------------------")
    print("Initializing mode assignment...")
    print(f"Model: {model}")
    print(f"Data file: {data}")
    print(f"Prompt file: {prompt_file}")
    print(f"Output file: {out_file}")
    print(f"Mode file: {mode_file}")
    print("-------------------")

    # Load and preprocess data
    df = pd.read_json(data, lines=True)
    df = record_formatting(df)
    records = df["record"].tolist()

    # Load prompt and modes list
    assignment_prompt = open(prompt_file, "r", encoding="utf-8").read()
    modes_root = ModeTree().from_mode_list(mode_file, from_file=True)

    # Annotate records
    responses = annotate_records(
        api_client,
        modes_root,
        records,
        assignment_prompt,
        temperature,
        max_workers,
    )

    # Save results
    df["responses"] = responses
    json_str = df.to_json(lines=True, orient='records', force_ascii=False)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(json_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use",
        default="gpt-4o"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/input/records.jsonl",
        help="Input JSONL file to run assignment on",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="data/output/files/records_annotated.jsonl",
        help="Output JSONL file to write results to",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompt/assignment.txt",
        help="File to read prompts from",
    )
    parser.add_argument(
        "--mode_file",
        type=str,
        default="data/output/modes/final_DRAST.md",
        help="Input markdown file to read modes from",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Maximum number of parallel threads for mode assignment"
    )

    args = parser.parse_args()
    assign_modes(
        args.model,
        args.data,
        args.out_file,
        args.prompt_file,
        args.mode_file,
        args.max_workers
    )
