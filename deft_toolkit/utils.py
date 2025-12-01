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


import regex
import time
import os
from anytree import Node
import traceback
from openai import OpenAI
import tiktoken
from typing import Tuple, List, Optional, Union

# Global API configuration
_global_config = {
    "llm_api_key": None,
    "llm_base_url": None,
    "embedding_api_key": None,
    "embedding_base_url": None,
}


def set_api_config() -> None:
    """
    Set global API configuration from environment variables.

    Required environment variables:
    - DEFT_LLM_API_KEY: The API key for LLM service.
    - DEFT_LLM_BASE_URL: The base URL for LLM API.
    - DEFT_EMBEDDING_API_KEY: The API key for embedding service.
    - DEFT_EMBEDDING_BASE_URL: The base URL for embedding API.
    """
    _global_config["llm_api_key"] = os.environ.get("API_KEY")
    _global_config["llm_base_url"] = os.environ.get("BASE_URL")
    _global_config["embedding_api_key"] = os.environ.get("API_KEY")
    _global_config["embedding_base_url"] = os.environ.get("BASE_URL")


def get_llm_config() -> Tuple[str, str]:
    """
    Get LLM API configuration from global config.

    Returns:
    - tuple: A tuple containing the LLM API key and base URL.
    """
    if _global_config["llm_api_key"] is None or _global_config["llm_base_url"] is None:
        raise ValueError(
            "LLM API configuration is not set."
        )
    return _global_config["llm_api_key"], _global_config["llm_base_url"]


def get_embedding_config() -> Tuple[str, str]:
    """
    Get embedding API configuration from global config.

    Returns:
    - tuple: A tuple containing the embedding API key and base URL.
    """
    if _global_config["embedding_api_key"] is None or _global_config["embedding_base_url"] is None:
        raise ValueError(
            "Embedding API configuration is not set."
        )
    return _global_config["embedding_api_key"], _global_config["embedding_base_url"]


class APIClient:
    """
    Client for interacting with OpenAI-compatible APIs.

    Attributes:
    - model (str): The model identifier used for API requests.
    - client (OpenAI): The OpenAI client instance for API interactions.

    Methods:
    - estimate_token_count: Estimate the number of tokens in a given prompt using Tiktoken.
    - truncating: Truncate a record to ensure it does not exceed a specified token limit.
    - iterative_prompt: Send a chat completion request to the API with automatic retry logic.
    """

    def __init__(self, model: str, llm_api_key: Optional[str] = None, llm_base_url: Optional[str] = None) -> None:
        """
        Initialize the API client for LLM interaction.

        Parameters:
        - model (str): The model identifier.
        - llm_api_key (str, optional): The API key for LLM service.
        - llm_base_url (str, optional): The base URL for LLM API.
        """
        self.model = model

        if llm_api_key is None or llm_base_url is None:
            llm_api_key, llm_base_url = get_llm_config()

        self.client = OpenAI(api_key=llm_api_key, base_url=llm_base_url)

    def estimate_token_count(self, prompt: str) -> int:
        """
        Estimate the number of tokens in a given prompt using Tiktoken.

        Parameters:
        - prompt (str): The input text to tokenize.

        Returns:
        - int: The estimated number of tokens.
        """
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")

        token_count = len(enc.encode(prompt))
        return token_count

    def truncating(self, record: str, max_tokens: int) -> str:
        """
        Truncate a record to ensure it does not exceed the specified token limit.

        Parameters:
        - record (str): The full record.
        - max_tokens (int): The maximum allowed number of tokens.

        Returns:
        - str: The truncated record.
        """
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except KeyError:
            print("Warning: model not found. Using o200k_base encoding.")
            enc = tiktoken.get_encoding("o200k_base")

        tokens = enc.encode(record)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return enc.decode(tokens)

    def iterative_prompt(self, prompt: str, temperature: float, num_try: int = 3) -> str:
        """
        Send a chat completion request to the API with retry logic.

        Parameters:
        - prompt (str): The input prompt.
        - temperature (float): Sampling temperature.
        - num_try (int): Number of retry attempts on failure. Defaults to 3.

        Returns:
        - str: The model's response content, or an empty string if all retries fail.
        """
        for attempt in range(num_try):
            try:
                message = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt},
                ]

                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=message,
                    temperature=temperature,
                    timeout=240,
                )
                return completion.choices[0].message.content

            except Exception as e:
                print(f"Attempt {attempt + 1}/{num_try} failed: {e}")
                if attempt < num_try - 1:
                    print("Waiting 20 seconds before next retry...")
                    time.sleep(20)
                else:
                    print("Max retries reached. Returning empty result.")
                    return ""


class ModeTree:
    """
    Represents a hierarchical tree structure for organizing modes.

    Attributes:
    - root (anytree.Node): The root node of the tree at level 0.
    - level_nodes (dict[int, anytree.Node]): A dictionary mapping level numbers to their corresponding nodes.

    Methods:
    - node_to_str: Convert a node to a formatted string representation.
    - from_mode_list: Construct a ModeTree from a list of mode strings or a file.
    - from_seed_file: Construct a ModeTree from a seed file without descriptions or counts.
    - _add_node: Add a node to the tree and merge with duplicates if present.
    - _remove_node_by_name_lvl: Remove a node from the tree by its name and level.
    - to_prompt_view: Generate a string representation of the tree with level-based indentation.
    - find_duplicates: Find all nodes with the same name and level in the tree.
    - to_file: Save the tree structure to a file.
    - to_mode_list: Convert the tree to a list of formatted mode strings.
    - get_root_descendants_name: Get the names of all descendant nodes.
    - update_tree: Update the tree by merging multiple modes into a new mode.
    - get_leaf_nodes_with_path: Get all leaf nodes along with their full paths from root.
    """

    def __init__(self, root_name: str = "Modes") -> None:
        """
        Initialize a ModeTree with a root node.

        Parameters:
        - root_name (str): The name of the root node. Defaults to "Modes".
        """
        self.root = Node(name=root_name, lvl=0, count=1, desc="Root mode", parent=None)
        self.level_nodes = {0: self.root}

    @staticmethod
    def node_to_str(node: Node, count: bool = True, desc: bool = True) -> str:
        """
        Convert a node to a string representation.

        Parameters:
        - node (Node): The node to convert.
        - count (bool): Whether to include count in the string.
        - desc (bool): Whether to include description in the string.

        Returns:
        - str: The string representation of the node.
        """
        if not count and not desc:
            return f"[{node.lvl}] {node.name}"
        elif not count and desc:
            return f"[{node.lvl}] {node.name}: {node.desc}"
        elif count and not desc:
            return f"[{node.lvl}] {node.name} (Count: {node.count})"
        else:
            return f"[{node.lvl}] {node.name} (Count: {node.count}): {node.desc}"

    @staticmethod
    def from_mode_list(mode_src: Union[List[str], str], from_file: bool = False) -> "ModeTree":
        """
        Construct a ModeTree from a list of mode strings or a file.

        Parameters:
        - mode_src (Union[List[str], str]): A list of mode strings or path to a file.
        - from_file (bool): Flag to indicate if the source is a file path.

        Returns:
        - ModeTree: The constructed ModeTree instance.
        """
        tree = ModeTree()
        mode_list = open(mode_src, "r", encoding='utf-8').readlines() if from_file else mode_src
        mode_list = [mode for mode in mode_list if len(mode.strip()) > 0]
        # Pattern to parse: [level] name (Count: n): description
        pattern = regex.compile(r"^\[(\d+)\] (.+?)(?: \(Count: (\d+)\))?(?::\s*(.+))?$")

        for mode in mode_list:
            if not mode.strip():
                continue
            try:
                match = regex.match(pattern, mode.strip())

                if not match:
                    print(f"Error reading: {mode.strip()}")
                    continue

                lvl, label, count, desc = (
                    int(match.group(1)),
                    match.group(2).strip(),
                    int(match.group(3)) if match.group(3) else 1,
                    match.group(4).strip() if match.group(4) else "",
                )

                tree._add_node(lvl, label, count, desc, tree.level_nodes.get(lvl - 1))

            except:
                print(match)
                print("Error reading", mode)
                traceback.print_exc()

        return tree

    @staticmethod
    def from_seed_file(seed_file: str) -> "ModeTree":
        """
        Construct a ModeTree from a seed file (no description/count).

        Parameters:
        - seed_file (str): The path to the seed file.

        Returns:
        - ModeTree: The constructed ModeTree instance.
        """
        tree = ModeTree()
        mode_list = open(seed_file, "r", encoding="utf-8").readlines() if seed_file else []
        mode_list = [mode for mode in mode_list if len(mode.strip()) > 0]
        pattern = regex.compile(r"^\[(\d+)\] (.+)")

        for mode in mode_list:
            if not mode.strip():
                continue
            try:
                match = regex.match(pattern, mode.strip())
                lvl, label = (
                    int(match.group(1)),
                    match.group(2).strip(),
                )
            except:
                print(match)
                print("Error reading", mode)
                traceback.print_exc()

            tree._add_node(lvl, label, 1, "", tree.level_nodes.get(lvl - 1))

        return tree

    def _add_node(self, lvl: int, label: str, count: int, desc: str, parent_node: Optional[Node]) -> None:
        """
        Add a node to the tree, merging with duplicates if present.

        Parameters:
        - lvl (int): The level of the node.
        - label (str): The name of the node.
        - count (int): The count of the node.
        - desc (str): The description of the node.
        - parent_node (Optional[Node]): The parent node of the new node.
        """
        if parent_node:
            # Check if node with same name already exists
            existing = next((n for n in parent_node.children if n.name == label), None)
            if existing:
                # Merge counts for duplicates
                existing.count += count
            else:
                new_node = Node(
                    name=label, lvl=lvl, count=count, desc=desc, parent=parent_node
                )
                self.level_nodes[lvl] = new_node

    def _remove_node_by_name_lvl(self, name: str, lvl: int) -> None:
        """
        Remove a node by name and level.

        Parameters:
        - name (str): The name of the node to remove.
        - lvl (int): The level of the node to remove.
        """
        node = next(
            (n for n in self.root.descendants if n.name == name and n.lvl == lvl), None
        )
        if node:
            node.parent = None

    def to_prompt_view(self, desc: bool = True) -> str:
        """
        Generate a string representation of the tree with indentation by level.

        Parameters:
        - desc (bool): Whether to include description in the string.

        Returns:
        - str: The string representation of the tree with indentation.
        """

        def traverse(node, result=""):
            if node.lvl > 0:
                result += (
                        "\t" * (node.lvl - 1)
                        + self.node_to_str(node, count=False, desc=desc)
                        + "\n"
                )
            for child in node.children:
                result = traverse(child, result)
            return result

        return traverse(self.root)

    def find_duplicates(self, name: str, level: int) -> List[Node]:
        """
        Find nodes with the same name and level in the tree.

        Parameters:
        - name (str): The name of the node to search for.
        - level (int): The level of the node to search for.

        Returns:
        - List[Node]: A list of nodes with the same name and level.
        """
        return [
            node
            for node in self.root.descendants
            if node.name.lower() == name.lower() and node.lvl == level
        ]

    def to_file(self, fname: str) -> None:
        """
        Save the tree to a file (only nodes with descriptions).

        Parameters:
        - fname (str): The path to the output file.
        """
        with open(fname, "w", encoding='utf-8') as f:
            for node in self.root.descendants:
                if len(node.desc) > 0:
                    indentation = "    " * (node.lvl - 1)
                    f.write(indentation + self.node_to_str(node) + "\n")

    def to_mode_list(self, desc: bool = True, count: bool = True) -> List[str]:
        """
        Convert the tree to a list of mode strings.

        Parameters:
        - desc (bool): Whether to include description in the string.
        - count (bool): Whether to include count in the string.

        Returns:
        - List[str]: A list of mode strings.
        """
        return [self.node_to_str(node, count, desc) for node in self.root.descendants]

    def get_root_descendants_name(self) -> List[str]:
        """
        Get the names of all descendant nodes.

        Returns:
        - List[str]: A list of all descendants' names.
        """
        return [node.name for node in self.root.descendants]

    def update_tree(self, original_modes: List[Tuple[str, int]], new_mode_name: str, new_mode_desc: str) -> "ModeTree":
        """
        Update the mode tree by merging a set of modes into a new mode.

        Parameters:
        - original_modes (List[Tuple[str, int]]): A list of tuples containing the name and level of modes to merge.
        - new_mode_name (str): The name of the new merged mode.
        - new_mode_desc (str): The description for the new merged mode.

        Returns:
        - ModeTree: The updated ModeTree instance (self).
        """
        total_count = 0
        parent_node = None
        nodes_to_merge = []

        # Collect all nodes to merge and calculate total count
        for name, lvl in original_modes:
            duplicates = self.find_duplicates(name, lvl)
            nodes_to_merge.extend(duplicates)
            total_count += sum(node.count for node in duplicates)
            if duplicates and not parent_node:
                parent_node = duplicates[0].parent

        if parent_node is None:
            parent_node = self.root

        # Create new merged node
        self._add_node(
            lvl=parent_node.lvl + 1,
            label=new_mode_name,
            count=total_count,
            desc=new_mode_desc,
            parent_node=parent_node,
        )

        # Remove old nodes
        for node in nodes_to_merge:
            if node.parent:
                node.parent = None

        return self

    def get_leaf_nodes_with_path(self) -> List[Tuple[Node, str]]:
        """
        Get all leaf nodes (nodes with no children) along with their full path from root.

        Returns:
        - List[Tuple[Node, str]]: A list of tuples containing the leaf node and its path string.
        """
        leaves = []
        for node in self.root.descendants:
            if not node.children: 
                path_parts = []
                current = node
                # Traverse from leaf to root, building path
                while current.parent is not None:
                    path_parts.append(self.node_to_str(current, count=False, desc=False))
                    current = current.parent
                path_parts.reverse()
                path_str = " → ".join(path_parts)
                leaves.append((node, path_str))
        return leaves
