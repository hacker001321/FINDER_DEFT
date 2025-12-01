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

import multiprocessing
import json
import os
import argparse
import re
from functools import partial
from .io_utils import load_jsonl
from .api import call_model


prompt_template = """你会看到一篇研究报告，研究报告正文中会有一些对参考文献的引用。
正文中的引用可能以如下形式出现：
1. 一段文字+空格+数字，例如："李强基于收入、教育和职业构造了一个社会经济地位指数（SES），将社会划分为7个等级 15"
2. 一段文字+[（一个或多个)数字]，例如："李强基于收入、教育和职业构造了一个社会经济地位指数（SES），将社会划分为7个等级[15]"
3. 一段文字+[（一个或多个)数字†(一些行号等内容)]，例如："李强基于收入、教育和职业构造了一个社会经济地位指数（SES），将社会划分为7个等级[15†L10][5L23][7†summary][9summary]"
4. [引用来源](引用链接)，例如："根据[ChinaFile: A Guide to Social Class in Modern China](https://www.chinafile.com/reporting-opinion/media/guide-social-class-modern-china)'s分类，中国社会可分为九个阶层"

请从正文中找出**所有**引用了参考文献的地方，提取出(fact, ref_idx, url)三元组，提取的时候，注意以下事项：
1. 由于后续需要检验这些facts是否正确，你可能需要在引用的前后寻找一些上下文，以确保fact是完整可理解的，而不是简单的词组或短语
2. 如果一个fact引用了多个文献，那么它应该对应多个三元组，例如如果引用了2个文献，则应该是(fact, ref_idx_1, url_1)和(fact, ref_idx_2, url_2)
3. 对于第三种形式的引用，ref_idx仅考虑第一个数字部分，不考虑其他指示具体位置的内容；对于第四种形式的引用（即引用来源和链接直接出现在正文中）的情况，ref_idx统一设置为0
4. 如果正文中没有标出引用的具体位置（比如仅在文章结尾列出了参考文献列表，而没有在正文中标出），请返回空列表
5. **重要**：如果参考文献列表中没有提供URL链接，url字段应该设置为空字符串""，而不是"None"或"null"

你应该返回json列表格式，列表中的每一项是一个三元组，例如：
[
    {{
        "fact": "原文中的文本片段，注意中文引号要用全角, 英文引号前加单个反斜杠转义",
        "ref_idx": "该段文字引用的参考文献在参考文献列表中的索引",
        "url": "该段文字引用的参考文献链接（从研究报告结尾的参考文献列表或引用处的括号中提取）"
    }}
]

下面是研究报告的正文：
{report_text}

下面开始提取，直接输出json列表，不要输出任何闲聊或解释。"""


prompt_template_en = """You will be provided with a research report. The body of the report will contain some citations to references.

Citations in the main text may appear in the following forms:
1. A segment of text + space + number, for example: "Li Qiang constructed a socioeconomic status index (SES) based on income, education, and occupation, dividing society into 7 levels 15"
2. A segment of text + [number], for example: "Li Qiang constructed a socioeconomic status index (SES) based on income, education, and occupation, dividing society into 7 levels[15]"
3. A segment of text + [number†(some line numbers, etc.)], for example: "Li Qiang constructed a socioeconomic status index (SES) based on income, education, and occupation, dividing society into 7 levels[15†L10][5L23][7†summary]"
4. [Citation Source](Citation Link), for example: "According to [ChinaFile: A Guide to Social Class in Modern China](https://www.chinafile.com/reporting-opinion/media/guide-social-class-modern-china)'s classification, Chinese society can be divided into nine strata"

Please identify **all** instances where references are cited in the main text, and extract (fact, ref_idx, url) triplets. When extracting, pay attention to the following:
1. Since these facts will need to be verified later, you may need to look for some context before and after the citation to ensure that the fact is complete and understandable, rather than just a simple phrase or short expression.
2. If a fact cites multiple references, then it should correspond to two triplets: (fact, ref_idx_1, url_1) and (fact, ref_idx_2, url_2).
3. For the third form of citation (i.e., where the citation source and link appear directly in the text), the ref_idx should be uniformly set to 0.
4. If the main text does not specify the exact location of the citation (for example, only the reference list is listed at the end of the article, without specifying the citation point in the text), please return an empty list.
5. **Important**: If the reference list does not provide a URL link, the url field should be set to an empty string "", not "None" or "null".

You should return a JSON list format, where each item in the list is a triplet, for example:
[
    {{
        "fact": "Text segment from the original document. Note that Chinese quotation marks should use full-width marks. And add a single backslash before the English quotation mark to make it a readable for python json module.",
        "ref_idx": "The index of the cited reference in the reference list for this text segment.",
        "url": "The URL of the cited reference for this text segment (extracted from the reference list at the end of the research report or from the parentheses at the citation point)."
    }}
]

Here is the main text of the research report:
{report_text}

Please begin the extraction now. Output only the JSON list directly, without any chitchat or explanations."""


def clean_urls(input_text):
    # match [title](url) format
    pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    
    def repl(match):
        title = match.group(1)
        url = match.group(2)
        # truncate #:~:text= and its content
        cut_idx = url.find('#:~:text=')
        if cut_idx != -1:
            url = url[:cut_idx]
        return f'[{title}]({url})'
    
    return pattern.sub(repl, input_text)


def remove_urls(input_text):
    # match [title](url) format, only remove the content in the parentheses, keep [title]
    pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    # replace [title](url) with [title]
    return pattern.sub(r'[\1]', input_text)


def clean_escape(input_text):
    """
    Clean invalid escape sequences in JSON strings.
    JSON only allows: \\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX
    All other \\x sequences are invalid and need to be fixed.
    """
    import re
    
    # First, handle common specific cases
    input_text = input_text.replace("\\>", ">")
    input_text = input_text.replace("\\<", "<")
    input_text = input_text.replace("\\+", "+")
    input_text = input_text.replace("\\~", "~")
    
    # Use regex to find and fix all invalid escape sequences
    # Valid escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    # Pattern to match backslash followed by any character that's NOT a valid escape
    def replace_invalid_escape(match):
        escaped_char = match.group(1)
        # If it's not a valid JSON escape character, remove the backslash
        valid_escapes = {'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'}
        if escaped_char not in valid_escapes:
            return escaped_char  # Remove the backslash
        else:
            return match.group(0)  # Keep valid escapes as-is
    
    # Match backslash followed by any single character (but not valid escapes)
    # This pattern finds \X where X is not a valid escape character
    pattern = r'\\([^"\\\/bfnrtu])'
    input_text = re.sub(pattern, replace_invalid_escape, input_text)
    
    return input_text


def run(data, output_path, id_to_lang_map):
    for i, d in enumerate(data):
        # Determine the language based on the id in the query data
        article_id = d.get('id')
        
        if not article_id:
            print(f"Article has no ID field, skipping extraction")
            continue
            
        if article_id not in id_to_lang_map:
            print(f"Language not found for article ID: {article_id}")
            continue
            
        lang = id_to_lang_map[article_id]
        if lang not in ["zh", "en"]:
            print(f"Unsupported language: {lang} for article ID: {article_id}")
            continue
            
        # Select the prompt based on the language
        if lang == "zh":
            user_prompt = prompt_template.format(report_text=d['article'])
        elif lang == 'en':
            user_prompt = prompt_template_en.format(report_text=d['article'])
        
        response = call_model(user_prompt)
        
        retries = 0
        while retries < 3:
            retries += 1
            try:
                if response != "":
                    response = response.replace("```json", "").replace("```", "")
                    response = clean_escape(response)
                    
                    d['citations'] = json.loads(response)

                    for c in d['citations']:
                        c['fact'] = remove_urls(c['fact'])

                else:
                    d['citations'] = "extraction failed"

                with open(output_path, "a+", encoding='utf-8') as f:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")

                print(f">>>>>>>>>> generating {d['id']}-th instance...")
                break
            except Exception as e:
                print(repr(e))
                continue
        
        if 'citations' not in d:
            print(f">>>>>>>>>> All attempts failed, article ID: {article_id}, cannot extract citations")


if __name__ == '__main__':

    multiprocessing.set_start_method('fork')

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--raw_data_path", type=str, required=True)
    parser.add_argument("--query_data_path", type=str, required=True, help="Path to query data with language information")
    parser.add_argument("--n_total_process", type=int, default=1)
    args = parser.parse_args()

    output_path = args.output_path
    
    # Load the query data to get language information
    query_data = load_jsonl(args.query_data_path)
    
    # Create a mapping from ID to language
    id_to_lang_map = {item['id']: item.get('language') for item in query_data if 'id' in item and 'language' in item}
    
    if not id_to_lang_map:
        raise ValueError("No valid language information found in query data")
    
    # Load the raw data
    raw_data = load_jsonl(args.raw_data_path)

    # If the output file exists, load the processed ids and filter out the processed instances
    if os.path.exists(output_path):
        processed = [d['id'] for d in load_jsonl(output_path)]
        data_to_process = [d for d in raw_data if d['id'] not in processed]
    else:
        data_to_process = raw_data

    # OpenAI deep research will add webpage snippets to the citations.
    # For fair comparison, we remove these snippets
    if 'openai' in args.raw_data_path:
        for d in data_to_process:
            d['article'] = clean_urls(d['article'])

    print(f"Processing {len(data_to_process)} instances...")

    n_total_process = args.n_total_process
    if n_total_process == 1:
        run(data_to_process, output_path, id_to_lang_map)
    elif n_total_process > 1:
        part_size = (len(data_to_process) + n_total_process - 1) // n_total_process
        data_splits = [data_to_process[i * part_size : (i + 1) * part_size] for i in range(n_total_process)]
        run_partial = partial(run, output_path=output_path, id_to_lang_map=id_to_lang_map)
        with multiprocessing.Pool(processes=n_total_process) as pool:
            results = pool.map(run_partial, data_splits)