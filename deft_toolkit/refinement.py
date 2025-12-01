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



from deft_toolkit.utils import ModeTree, APIClient, get_embedding_config
import argparse
import time
import traceback
import regex
import numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


def mode_pairs(
    mode_sent: List[str],
    all_pairs: List[List[str]],
    merge_threshold: float,
    embedding_api_key: Optional[str] = None,
    embedding_base_url: Optional[str] = None
) -> Tuple[List[str], List[List[str]]]:
    """
    Identify the most similar mode pair above the similarity threshold that hasn't been processed yet.

    Parameters:
    - mode_sent (List[str]): The list of mode sentences to compare.
    - all_pairs (List[List[str]]): The list of already processed mode pairs (each sorted).
    - merge_threshold (float): The cosine similarity threshold for merging (e.g., 0.6).
    - embedding_api_key (Optional[str]): The API key for embedding service. Uses global config if not provided.
    - embedding_base_url (Optional[str]): The base URL for embedding API. Uses global config if not provided.

    Returns:
    - Tuple[List[str], List[List[str]]]: A tuple containing the flattened list of the selected mode pair and the updated list of all processed pairs.
    """

    if embedding_api_key is None or embedding_base_url is None:
        embedding_api_key, embedding_base_url = get_embedding_config()

    client = OpenAI(
        api_key=embedding_api_key,
        base_url=embedding_base_url,
    )

    embeddings = None

    # Get embeddings from the API
    for attempt in range(3):
        try:
            resp = client.embeddings.create(
                model="doubao-embedding-large-text-250515",
                input=mode_sent,
                encoding_format="float"
            )
            embeddings = np.array([data.embedding for data in resp.data])
            break
        except Exception as e:
            if attempt < 2:
                wait_time = (attempt + 1) * 10
                print(f"[Embedding] Waiting {wait_time}s before retry...\n{e}")
                time.sleep(wait_time)
            else:
                print(f"[Embedding] All 3 attempts failed. Using vector of size (1, 2048), 0.5.\n{e}")

    if embeddings is None:
        embeddings = np.full((1, 2048), 0.5)

    # Compute pairwise cosine similarities
    cosine_scores = cosine_similarity(embeddings)

    # Generate and sort all unique mode pairs by similarity
    pairs = [
        {"index": [i, j], "score": cosine_scores[i][j].item()}
        for i in range(len(cosine_scores))
        for j in range(i + 1, len(cosine_scores))
    ]
    pairs = sorted(pairs, key=lambda x: x["score"], reverse=True)

    # Select the first valid pair above threshold and not previously processed
    selected_pairs = []
    for pair in pairs:
        if len(selected_pairs) >= 1:
            break
        i, j = pair["index"]
        if (
                pair["score"] > merge_threshold
                and sorted([mode_sent[i], mode_sent[j]]) not in all_pairs
        ):
            selected_pairs.append([mode_sent[i], mode_sent[j]])
            all_pairs.append(sorted([mode_sent[i], mode_sent[j]]))

    flattened = [item for sublist in selected_pairs for item in sublist]

    return flattened, all_pairs


def merge_modes(
    modes_root: ModeTree,
    refinement_prompt: str,
    api_client: APIClient,
    temperature: float,
    merge_threshold: float,
    embedding_api_key: Optional[str] = None,
    embedding_base_url: Optional[str] = None
) -> ModeTree:
    """
    Iteratively merge semantically similar modes using an LLM until no more valid pairs exist.

    Parameters:
    - modes_root (ModeTree): The root of the mode hierarchy tree.
    - refinement_prompt (str): The prompt template for merging modes.
    - api_client (APIClient): The API client for LLM service.
    - temperature (float): The sampling temperature for LLM generation.
    - merge_threshold (float): The similarity threshold for merging.
    - embedding_api_key (Optional[str]): The API key for embedding service. Uses global config if not provided.
    - embedding_base_url (Optional[str]): The base URL for embedding API. Uses global config if not provided.

    Returns:
    - ModeTree: The updated mode tree after merging.
    """

    orig_new = {}

    # Precompile regex patterns for parsing LLM output
    pattern_mode = regex.compile(r"^\[(\d+)\]([^:]+?):(.+?)\(([^)]+)\)$")
    pattern_original = regex.compile(r"\[(\d+)\]([^,\)]+)")

    all_pairs = []

    # Keep merging while valid similar mode pairs exist
    while True:
        mode_sent = modes_root.to_mode_list(desc=True, count=False)
        new_pairs, all_pairs = mode_pairs(mode_sent, all_pairs, merge_threshold, embedding_api_key, embedding_base_url)

        if len(new_pairs) < 2:
            print("No more mode pairs to merge.")
            break

        print("Modes sent to model:")
        for i, mode in enumerate(new_pairs):
            print(f"   {i + 1}. {mode}")

        try:
            # Format prompt and call LLM to generate merged mode
            refiner_prompt_formatted = refinement_prompt.format(Modes="\n".join(new_pairs))
            response = api_client.iterative_prompt(refiner_prompt_formatted, temperature)
            response = response.strip()
            response = response.replace("（", "(").replace("）", ")")
            print(f"Model output: {response}")

            if not response or response.lower() == "none":
                print("Empty or 'None' response, skipping...")
                continue

            merges = [line.strip() for line in response.split("\n") if line.strip()]
            merged_any = False

            # Parse each merge to update the mode tree and track renaming
            for merge in merges:
                match = pattern_mode.match(merge)
                if not match:
                    print("Parsing failure based on regular expression 1, skipping...")
                    continue

                try:
                    lvl = int(match.group(1).strip())
                    name = match.group(2).strip()
                    desc = match.group(3).strip()
                    originals_list = match.group(4).strip()

                    orig_matches = pattern_original.findall(originals_list)
                    if not orig_matches:
                        print("Parsing failure based on regular expression 2, skipping...")
                        continue

                    original_modes = [
                        (t[1].strip(), int(t[0])) for t in orig_matches
                    ]

                    modes_root = modes_root.update_tree(original_modes, name, desc)

                    for orig_name, _ in original_modes:
                        orig_new[orig_name] = name
                    print(f"Merged into [{lvl}] {name}: {desc}")
                    merged_any = True

                except Exception as e:
                    print(f"Error processing merge line: {merge}\n {e}")
                    traceback.print_exc()

            if not merged_any:
                print("Parsing and merging failed.")

        except Exception as e:
            print("Error when calling LLM!")
            traceback.print_exc()

    final_mode_list = modes_root.to_mode_list(desc=True)
    print(f"Number of modes after Stage 1: {len(final_mode_list)}")
    print(f"Mapping updated: {orig_new}")

    return modes_root


def remove_modes(
    modes_root: ModeTree,
    remove_threshold: float
) -> ModeTree:
    """
    Remove modes whose document count is below a frequency threshold.

    Parameters:
    - modes_root (ModeTree): The root of the mode hierarchy tree.
    - remove_threshold (float): The fraction of total count below which modes are removed.

    Returns:
    - ModeTree: The updated mode tree with low-frequency modes removed.
    """

    total_count = sum(node.count for node in modes_root.root.children)
    threshold_count = total_count * remove_threshold
    print(f"Total Count: {total_count}, Threshold Count: {threshold_count}.")
    removed = False

    # Remove modes with count below threshold
    for node in modes_root.root.children:
        if node.count < threshold_count and node.lvl == 1:
            node.parent = None
            print(f"Removing {node.name} ({node.count} counts)")
            removed = True

    if not removed:
        print("No modes are removed during Stage 2.")

    return modes_root


def refine_modes(
    model: str,
    prompt_file: str,
    mode_file: str,
    out_file: str,
    merge_threshold: float = 0.6,
    remove_threshold: float = 0.01,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
) -> None:
    """
    Refine modes through semantic merging and low-frequency pruning.

    Parameters:
    - model (str): The name of the LLM model to use.
    - prompt_file (str): The path to the refinement prompt template file.
    - mode_file (str): The path to the input mode file (Markdown format).
    - out_file (str): The path to the output refined mode file.
    - merge_threshold (float): The cosine similarity threshold for merging. Defaults to 0.6.
    - remove_threshold (float): The frequency ratio threshold for removal. Defaults to 0.01.
    - llm_api_key (Optional[str]): The API key for LLM service. Uses global config if not provided.
    - llm_base_url (Optional[str]): The base URL for LLM API. Uses global config if not provided.
    - embedding_api_key (Optional[str]): The API key for embedding service. Uses global config if not provided.
    - embedding_base_url (Optional[str]): The base URL for embedding API. Uses global config if not provided.
    """
    api_client = APIClient(model=model, llm_api_key=llm_api_key, llm_base_url=llm_base_url)
    temperature = 0.1
    modes_root = ModeTree().from_mode_list(mode_file, from_file=True)

    print("-------------------")
    print("Initializing mode refinement...")
    print(f"Model: {model}")
    print(f"Prompt file: {prompt_file}")
    print(f"Mode file: {mode_file}")
    print(f"Output file: {out_file}")
    print("-------------------")

    refinement_prompt = open(prompt_file, "r", encoding="utf-8").read()

    # Stage 1: Merge semantically similar modes
    updated_modes_root = merge_modes(
        modes_root,
        refinement_prompt,
        api_client,
        temperature,
        merge_threshold,
        embedding_api_key,
        embedding_base_url
    )

    # Stage 2: Remove low-frequency modes
    updated_modes_root = remove_modes(updated_modes_root, remove_threshold)

    updated_modes_root.to_file(out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompt/refinement.txt",
        help="File to read prompts from"
    )
    parser.add_argument(
        "--mode_file",
        type=str,
        default="data/output/modes/modes.md",
        help="Input markdown file to read modes from"
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="data/output/modes/refinement.md",
        help="Output markdown file to write refined modes to"
    )
    parser.add_argument(
        "--merge_threshold",
        type=float,
        default=0.6,
        help="Cosine similarity threshold for merging modes"
    )
    parser.add_argument(
        "--remove_threshold",
        type=float,
        default=0.01,
        help="Frequency threshold (ratio) for removing modes"
    )

    args = parser.parse_args()
    refine_modes(
        args.model,
        args.prompt_file,
        args.mode_file,
        args.out_file,
        args.merge_threshold,
        args.remove_threshold
    )


