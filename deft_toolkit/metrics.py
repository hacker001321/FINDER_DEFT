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

from deft_toolkit.utils import ModeTree
import argparse
import math
import re
from typing import Dict, Tuple
from collections import defaultdict
import pandas as pd


def compute_metrics(
    data: str,
    mode_file: str,
    output_col: str
) -> Tuple[Dict[str, float], float]:
    """
    Calculate Taxonomy Positive Metric (S) based on root-level mode frequency.

    Parameters:
    - data (str): The path to the JSONL file with annotations.
    - mode_file (str): The path to the mode tree file in Markdown format.
    - output_col (str): The column name containing model output.

    Returns:
    - Tuple[Dict[str, float], float]: A tuple containing S scores for each root mode and the average S score.
    """
    tree = ModeTree().from_mode_list(mode_file, from_file=True)
    leaf_with_path = tree.get_leaf_nodes_with_path()

    leaf_to_root = {}
    valid_leaf_names = set()

    for leaf_node, path_str in leaf_with_path:
        root_match = re.match(r"^\[1\]\s*([^\n→:]+)", path_str)
        if root_match:
            root_name = root_match.group(1).strip()
            leaf_to_root[leaf_node.name] = root_name
            valid_leaf_names.add(leaf_node.name)

    leaf_name_lower_to_canonical = {name.lower(): name for name in valid_leaf_names}

    root_modes = {node.name for node in tree.root.descendants if node.lvl == 1}

    df = pd.read_json(data, lines=True)

    # Count E_i for each root mode
    E_count = defaultdict(int)

    for pred_text in df[output_col]:
        if not isinstance(pred_text, str):
            continue

        predicted_leaf_names = []
        lines = pred_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            match = re.match(r'^\s*(?:\[\d+\]\s*)?([^(:\n]+?)(?:\s*\(level\s+\d+\))?\s*:', line, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                canonical = leaf_name_lower_to_canonical.get(candidate.lower())
                if canonical:
                    predicted_leaf_names.append(canonical)

        roots_in_doc = set()
        for name in predicted_leaf_names:
            if name in leaf_to_root:
                roots_in_doc.add(leaf_to_root[name])

        for root in roots_in_doc:
            E_count[root] += 1

    # Compute S_i and average S
    S_scores = {}
    total_S = 0.0
    for root in root_modes:
        E = E_count[root]
        ratio = min(E / 100.0, 1.0)
        S_i = 100.0 * math.cos(ratio * math.pi / 2)
        S_scores[root] = S_i
        total_S += S_i

    avg_S = total_S / len(root_modes) if root_modes else 0.0

    print("\n" + "=" * 70)
    print("S_i = 100 * cos( (E_i / 100) * π/2 )")
    print("E_i = Number of records exhibiting root mode i")
    print("=" * 70)
    for root in sorted(S_scores.keys()):
        print(f"{root:<30} | E={E_count[root]:>4} | S={S_scores[root]:>8.4f}")
    print("-" * 70)
    print(f"{'Average S':<30} |        | {avg_S:>8.4f}")
    print("=" * 70)

    return S_scores, avg_S


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate alignment metrics between modes and ground-truth modes."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/output/files/records_annotated.jsonl",
        help="Input JSONL file containing both ground-truth and predicted modes",
    )
    parser.add_argument(
        "--mode_file",
        type=str,
        default="data/output/modes/final_DRAST.md",
        help="Input markdown file containing mode hierarchy")

    parser.add_argument(
        "--output_col",
        type=str,
        default="responses",
        help="Column name for predicted modes",
    )
    args = parser.parse_args()

    compute_metrics(
        args.data,
        args.mode_file,
        args.output_col
    )
