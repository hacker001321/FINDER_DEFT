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

import os
import argparse
from utils import load_jsonl
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    total_citations = 0
    total_valid_citations = 0
    total_num = 0
    total_usage = [0, 0]

    data = load_jsonl(args.input_path)

    for d in tqdm(data):
        if not d['citations']:
            continue
        for c in d['citations_deduped'].values():
            if c['validate_error'] is not None:
                continue
            for _c in c['validate_res']:
                if _c['result'] != 'unknown':
                    total_citations += 1
                    if _c['result'] == 'supported':
                        total_valid_citations += 1


        total_num += 1



    with open(args.output_path, 'w') as f:
        f.write(f'total_citations: {total_citations/total_num}\n')
        f.write(f'total_valid_citations: {total_valid_citations/total_num}\n')
        f.write(f'valid_rate: {total_valid_citations / total_citations}\n')