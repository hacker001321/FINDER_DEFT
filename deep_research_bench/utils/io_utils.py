# coding=utf-8
# Copyright 2024 deep_research_bench Inc.
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

# Portions of this file are modifications by OPPO PersonalAI Team.
# Licensed under the Apache License, Version 2.0.

import json
import re

def clean_json_escape(input_text):
    """
    Clean invalid escape sequences in JSON strings.
    JSON only allows: \\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX
    All other \\x sequences are invalid and need to be fixed.
    
    Args:
        input_text (str): JSON string that may contain invalid escape sequences
        
    Returns:
        str: Cleaned JSON string with invalid escapes removed
    """
    if not isinstance(input_text, str):
        return input_text
    
    # First, handle common specific cases
    input_text = input_text.replace("\\>", ">")
    input_text = input_text.replace("\\<", "<")
    input_text = input_text.replace("\\+", "+")
    input_text = input_text.replace("\\~", "~")
    input_text = input_text.replace("\\-", "-")
    input_text = input_text.replace("\\=", "=")
    input_text = input_text.replace("\\*", "*")
    input_text = input_text.replace("\\&", "&")
    input_text = input_text.replace("\\%", "%")
    input_text = input_text.replace("\\$", "$")
    input_text = input_text.replace("\\#", "#")
    input_text = input_text.replace("\\@", "@")
    input_text = input_text.replace("\\!", "!")
    input_text = input_text.replace("\\?", "?")
    input_text = input_text.replace("\\(", "(")
    input_text = input_text.replace("\\)", ")")
    input_text = input_text.replace("\\[", "[")
    input_text = input_text.replace("\\]", "]")
    input_text = input_text.replace("\\{", "{")
    input_text = input_text.replace("\\}", "}")
    input_text = input_text.replace("\\|", "|")
    input_text = input_text.replace("\\:", ":")
    input_text = input_text.replace("\\;", ";")
    input_text = input_text.replace("\\,", ",")
    input_text = input_text.replace("\\'", "'")
    input_text = input_text.replace("\\`", "`")
    
    # Use regex to find and fix all remaining invalid escape sequences
    # Valid escapes in JSON: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    def replace_invalid_escape(match):
        escaped_char = match.group(1)
        # If it's not a valid JSON escape character, remove the backslash
        valid_escapes = {'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'}
        if escaped_char not in valid_escapes:
            return escaped_char  # Remove the backslash
        else:
            return match.group(0)  # Keep valid escapes as-is
    
    # Match backslash followed by any single character (but not valid escapes)
    pattern = r'\\([^"\\\/bfnrtu])'
    input_text = re.sub(pattern, replace_invalid_escape, input_text)
    
    return input_text

def safe_json_loads(json_str):
    """
    Safely parse JSON string by cleaning invalid escape sequences first.
    
    Args:
        json_str (str): JSON string to parse
        
    Returns:
        dict/list: Parsed JSON object
        
    Raises:
        json.JSONDecodeError: If JSON is still invalid after cleaning
    """
    cleaned_str = clean_json_escape(json_str)
    return json.loads(cleaned_str)

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    return data 