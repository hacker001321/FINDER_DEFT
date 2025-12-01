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


from deft_toolkit.utils import APIClient, ModeTree
from tqdm import tqdm
import regex
import traceback
import argparse
import re
from urllib.parse import urldefrag
import random
from typing import List, Tuple, Optional
import pandas as pd


def prompt_formatting(
        generation_prompt: str,
        api_client: APIClient,
        report: str,
        modes_list: List[str],
        context_len: int,
) -> str:
    """
    Format the LLM prompt by inserting the report and modes with truncation if needed.

    Parameters:
    - generation_prompt (str): The prompt template with {Report} and {Modes} placeholders.
    - api_client (APIClient): The API client for LLM service.
    - report (str): The input report text.
    - modes_list (List[str]): A list of seed mode strings.
    - context_len (int): The maximum allowed context length in tokens.

    Returns:
    - str: The final formatted prompt ready for the LLM.
    """

    # Shuffle modes to avoid ordering bias
    random.shuffle(modes_list)
    mode_str = "\n".join(modes_list)

    # Remove URL fragments
    report = re.sub(r'https?://[^\s\)]+', lambda m: urldefrag(m.group(0))[0], report)

    # Estimate token counts for each component
    report_len = api_client.estimate_token_count(report)
    prompt_len = api_client.estimate_token_count(generation_prompt)
    mode_len = api_client.estimate_token_count(mode_str)
    total_len = prompt_len + report_len + mode_len

    # Truncate report if total tokens exceed context limit
    if total_len > context_len:
        print(f"Report is too long ({report_len} tokens). Truncating...")
        report = api_client.truncating(report, context_len - prompt_len - mode_len)

    # Insert report and modes into the prompt template
    prompt = generation_prompt.format(Report=report, Modes=mode_str)
    return prompt


def report_formatting(
        df: pd.DataFrame,
        lang: Optional[str] = None,
) -> pd.DataFrame:
    """
    Construct a unified 'report' field from 'question', 'article', and 'failure_analysis' columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with 'question', 'article', and 'failure_analysis' columns.
    - lang (Optional[str]): The language to force ('en' or 'zh'). If None, auto-detects from 'question'.

    Returns:
    - pd.DataFrame: A copy of the input DataFrame with an added 'report' column.
    """

    df = df.copy()
    report_list = []
    for _, row in df.iterrows():
        # Safely extract and clean fields
        question = '' if pd.isna(row.get('question', '')) else str(row.get('question', '')).strip()
        article = '' if pd.isna(row.get('article', '')) else str(row.get('article', '')).strip()
        failure_analysis = '' if pd.isna(row.get('failure_analysis', '')) else str(row.get('failure_analysis', '')).strip()

        # Auto-detect language and format report based on language
        detected_lang = "zh" if re.search(r'[\u4e00-\u9fff]', question) else "en"
        use_lang = lang if lang is not None else detected_lang

        if use_lang == "zh":
            report = f"失效分析：\n{failure_analysis}\n问题：\n{question}\n答案：\n{article}"
        else:
            report = f"Failure Analysis: \n{failure_analysis}\nQuestion: \n{question}\narticle: \n{article}\n"
        report_list.append(report)

    df['report'] = report_list
    return df


def coding_reports(
        modes_root: ModeTree,
        modes_list: List[str],
        context_len: int,
        reports: List[str],
        api_client: APIClient,
        generation_prompt: str,
        temperature: float,
) -> Tuple[List[str], List[str], ModeTree]:
    """
    Generate modes from reports using an LLM and update the mode tree.

    Parameters:
    - modes_root (ModeTree): The root of the mode hierarchy tree.
    - modes_list (List[str]): The current list of seed mode strings.
    - context_len (int): The maximum context length in tokens for the model.
    - reports (List[str]): A list of input report strings.
    - api_client (APIClient): The API client for LLM service.
    - generation_prompt (str): The prompt template for mode generation.
    - temperature (float): The sampling temperature for LLM generation.

    Returns:
    - Tuple[List[str], List[str], ModeTree]: A tuple containing the responses, updated modes list, and updated mode tree.
    """

    responses = []
    mode_format = regex.compile(r"^\[(\d+)\]\s*([\w\s]+?)\s*[：:]\s*(.+)")

    for i, report in enumerate(tqdm(reports)):
        prompt = prompt_formatting(
            generation_prompt,
            api_client,
            report,
            modes_list,
            context_len,
        )

        try:
            response = api_client.iterative_prompt(prompt, temperature)

            # Parse modes and update mode tree
            modes = [t.strip() for t in response.split("\n")]
            for t in modes:
                if not regex.match(mode_format, t):
                    print(f"Invalid mode format: {t}. Skipping...")
                    continue
                groups = regex.match(mode_format, t)
                lvl, name, desc = int(groups[1]), groups[2].strip(), groups[3].strip()

                if lvl != 1:
                    print(f"Lower level modes are not allowed: {t}. Skipping...")
                    continue
                dups = modes_root.find_duplicates(name, lvl)

                if dups:
                    dups[0].count += 1
                else:
                    modes_root._add_node(lvl, name, 1, desc, modes_root.root)
                    modes_list = modes_root.to_mode_list(desc=False, count=False)

            print(f"Modes: {response}")
            print("--------------------")
            responses.append(response)

        except Exception as e:
            print(f"Skipping report {i} due to error: {e}")
            traceback.print_exc()
            responses.append("Error - Exception")
            continue

    return responses, modes_list, modes_root


def generate_modes(
        model: str,
        data: str,
        prompt_file: str,
        seed_file: str,
        out_file: str,
        mode_file: str,
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
) -> ModeTree:
    """
    Generate modes from input reports using a language model.

    Parameters:
    - model (str): The name of the LLM model to use.
    - data (str): The path to the input JSONL file containing reports.
    - prompt_file (str): The path to the prompt template file.
    - seed_file (str): The path to the seed modes markdown file.
    - out_file (str): The path to the output JSONL file for responses.
    - mode_file (str): The path to the output markdown file for generated modes.
    - llm_api_key (Optional[str]): The API key for LLM service. Uses global config if not provided.
    - llm_base_url (Optional[str]): The base URL for LLM API. Uses global config if not provided.

    Returns:
    - ModeTree: The root node of the updated mode hierarchy.
    """

    api_client = APIClient(model=model, llm_api_key=llm_api_key, llm_base_url=llm_base_url)
    max_tokens, temperature = 4096, 0.1

    print("-------------------")
    print("Initializing mode generation...")
    print(f"Model: {model}")
    print(f"Data file: {data}")
    print(f"Prompt file: {prompt_file}")
    print(f"Seed file: {seed_file}")
    print(f"Output file: {out_file}")
    print(f"Mode file: {mode_file}")
    print("-------------------")

    # Set context window based on model
    context = {
        "qwen3-max-preview": 256000,
        "grok-4-0709": 256000,
        "deepseek-v3.1": 128000,
        "claude-opus-4-1-20250805": 200000,
        "gemini-2.5-pro": 1000000,
    }.get(model, 128000)
    context_len = context - max_tokens

    # Load and preprocess data
    df = pd.read_json(data, lines=True)
    df = report_formatting(df)
    reports = df["report"].tolist()

    # Load prompt and seed modes
    with open(prompt_file, "r", encoding="utf-8") as f:
        generation_prompt = f.read()
    modes_root = ModeTree().from_seed_file(seed_file)
    modes_list = modes_root.to_mode_list(desc=True, count=False)

    # Generate modes from reports
    responses, modes_list, modes_root = coding_reports(
        modes_root,
        modes_list,
        context_len,
        reports,
        api_client,
        generation_prompt,
        temperature,
    )

    # Save updated mode tree
    modes_root.to_file(mode_file)

    # Save responses alongside original data
    df["responses"] = responses
    json_str = df.to_json(lines=True, orient='records', force_ascii=False)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(json_str)
    return modes_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use",
        default="gpt-4o")
    parser.add_argument(
        "--data",
        type=str,
        default="data/input/records_with_analysis.jsonl",
        help="Input JSONL file to run mode generation on",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="data/output/files/generation.jsonl",
        help="Output JSONL file to write results to",
    )
    parser.add_argument(
        "--mode_file",
        type=str,
        default="data/output/modes/modes.md",
        help="Output markdown file to write modes to",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompt/modes_generation.txt",
        help="Prompt template file",
    )
    parser.add_argument(
        "--seed_file",
        type=str,
        default="prompt/seeds.md",
        help="Seed modes markdown file",
    )


    args = parser.parse_args()
    generate_modes(
        args.model,
        args.data,
        args.out_file,
        args.mode_file,
        args.prompt_file,
        args.seed_file,
    )
