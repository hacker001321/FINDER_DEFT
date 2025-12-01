# Checklist Pass Rate Evaluation

This module evaluates deep research agent outputs by measuring their compliance with predefined checklist criteria.

## Overview

The checklist evaluator uses an LLM judge to assess whether agent-generated articles meet specific quality criteria. It provides:

- **Automated evaluation** of multiple articles in parallel
- **Detailed pass/fail reports** for each checklist item
- **Overall statistics** including pass rates and completion metrics
- **Retry logic** with exponential backoff for robust API handling

## Quick Start

### 1. Environment Setup

Set the required environment variables:

```bash
export API_KEY="your_openai_api_key"
export MODEL_NAME="gpt-4o"  # or your preferred evaluation model
export BASE_URL="https://api.openai.com/v1"  # or your custom endpoint
```

### 2. Prepare Data

#### Agent Results

Place your agent's inference results in the `data/` directory as `.jsonl` files. Each line should contain:

```json
{
  "id": "article_001",
  "article": "Your agent's generated article content..."
}
```

Or alternatively use `prediction` field:

```json
{
  "id": "article_001",
  "prediction": "Your agent's prediction content..."
}
```

#### Checklist Criteria

Ensure `data/checklist.jsonl` exists with evaluation criteria:

```json
{
  "id": "article_001",
  "topic": "Climate Change Impact on Agriculture",
  "checklist": [
    {
      "title": "Comprehensive Coverage",
      "description": "The article should cover at least 3 major aspects of climate change impact on agriculture"
    },
    {
      "title": "Data-Backed Claims",
      "description": "All major claims should be supported by specific data, statistics, or research findings"
    },
    {
      "title": "Regional Analysis",
      "description": "The article should discuss impacts across different geographical regions"
    }
  ]
}
```

### 3. Run Evaluation

**Basic usage** (with default settings):

```bash
python llm_judge_v2.py
```

**With custom parameters**:

```bash
python llm_judge_v2.py \
  --input_folder ./my_data \
  --output_folder ./my_results \
  --checklist_file ./data/my_checklist.jsonl \
  --max_concurrent_requests 20 \
  --request_timeout 180 \
  --max_retries 3
```

**View all options**:

```bash
python llm_judge_v2.py --help
```

Output:
```
usage: llm_judge_v2.py [-h] [--input_folder INPUT_FOLDER]
                       [--output_folder OUTPUT_FOLDER]
                       [--checklist_file CHECKLIST_FILE]
                       [--max_concurrent_requests MAX_CONCURRENT_REQUESTS]
                       [--request_timeout REQUEST_TIMEOUT]
                       [--max_retries MAX_RETRIES]

Checklist-based evaluation for deep research articles using LLM judge

optional arguments:
  -h, --help            show this help message and exit
  --input_folder INPUT_FOLDER
                        Input folder containing .jsonl files (default: ./data)
  --output_folder OUTPUT_FOLDER
                        Output folder for evaluation results (default: ./evaluation_checklist_results)
  --checklist_file CHECKLIST_FILE
                        Checklist criteria file (default: ./data/checklist.jsonl)
  --max_concurrent_requests MAX_CONCURRENT_REQUESTS
                        Maximum concurrent API requests (default: 10)
  --request_timeout REQUEST_TIMEOUT
                        API request timeout in seconds (default: 120)
  --max_retries MAX_RETRIES
                        Maximum retry attempts (default: 5)
```

### 4. View Results

Results are saved in `evaluation_checklist_results/` directory:

```
evaluation_checklist_results/
├── eval_agent_model_1.jsonl
├── eval_agent_model_2.jsonl
└── ...
```

Each result file contains:

```json
{
  "id": "article_001",
  "topic": "Climate Change Impact on Agriculture",
  "checklist_evaluations": [
    {
      "checklist_title": "Comprehensive Coverage",
      "is_met": true,
      "model_used": "gpt-4o"
    },
    {
      "checklist_title": "Data-Backed Claims",
      "is_met": false,
      "model_used": "gpt-4o"
    }
  ],
  "model_used": "gpt-4o"
}
```

## Configuration

### Environment Variables (Required)

Set these in your environment or `.env` file:

```bash
export API_KEY="your_openai_api_key"
export MODEL_NAME="gpt-4o"
export BASE_URL="https://api.openai.com/v1"
```

### Command Line Arguments (Optional)

All evaluation parameters can be configured via command line arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_folder` | `./data` | Input folder containing .jsonl files |
| `--output_folder` | `./evaluation_checklist_results` | Output folder for evaluation results |
| `--checklist_file` | `./data/checklist.jsonl` | Checklist criteria file path |
| `--max_concurrent_requests` | `10` | Maximum parallel API calls (reduce if hitting rate limits) |
| `--request_timeout` | `120` | API request timeout in seconds |
| `--max_retries` | `5` | Maximum retry attempts with exponential backoff |

**Example:**

```bash
# Use custom concurrency and timeout
python llm_judge_v2.py --max_concurrent_requests 20 --request_timeout 180

# Use custom data paths
python llm_judge_v2.py --input_folder /path/to/data --output_folder /path/to/results
```

## Features

### Async Processing

The evaluator uses asynchronous processing for efficiency:
- Concurrent evaluation of multiple articles
- Concurrent evaluation of checklist items within each article
- Configurable concurrency limits to respect API rate limits

### Robust Error Handling

- **Timeout handling**: Automatically retries timed-out requests
- **Exponential backoff**: Waits increasingly longer between retries
- **Graceful degradation**: Continues evaluation even if some items fail
- **Detailed logging**: Prints progress and error information

### Flexible Input

Supports multiple input formats:
- `article` field for generated articles
- `prediction` field for model predictions
- Processes all `.jsonl` files in the input directory
- Skips already-evaluated files

## Output Statistics

After evaluation completes, you'll see:

```
============================================================
All files processed!
============================================================
Files processed: 3
Total articles evaluated: 150
Successful checks: 1245/1500
Pass rate: 83.00%
Total time: 1234.56 seconds
Average per check: 0.82 seconds
```

## Example Workflows

### Workflow 1: Basic Evaluation (Default Settings)

```bash
# 1. Set environment variables
export API_KEY="sk-..."
export MODEL_NAME="gpt-4o"
export BASE_URL="https://api.openai.com/v1"

# 2. Place agent results in data/
cp /path/to/agent_results.jsonl data/

# 3. Ensure checklist criteria exists
ls data/checklist.jsonl

# 4. Run evaluation with defaults
python llm_judge_v2.py

# 5. Check results
ls evaluation_checklist_results/
cat evaluation_checklist_results/eval_agent_results.jsonl
```

### Workflow 2: Custom Configuration

```bash
# Run with custom settings for large-scale evaluation
python llm_judge_v2.py \
  --input_folder ./large_dataset \
  --output_folder ./results_batch_1 \
  --max_concurrent_requests 30 \
  --request_timeout 300 \
  --max_retries 10
```

### Workflow 3: Testing with Small Dataset

```bash
# Test with reduced concurrency for debugging
python llm_judge_v2.py \
  --input_folder ./test_data \
  --output_folder ./test_results \
  --max_concurrent_requests 2 \
  --request_timeout 60
```

## Troubleshooting

### Rate Limit Errors

Reduce concurrent requests to make fewer parallel calls:

```bash
python llm_judge_v2.py --max_concurrent_requests 5
```

### Timeout Issues

Increase timeout for slower APIs:

```bash
python llm_judge_v2.py --request_timeout 300  # 5 minutes
```

### Processing Large Datasets

For large datasets, increase concurrency and adjust timeout:

```bash
python llm_judge_v2.py \
  --max_concurrent_requests 30 \
  --request_timeout 240 \
  --max_retries 10
```

### Missing Dependencies

Install required packages:

```bash
pip install openai asyncio
```

### File Not Found

Ensure your directory structure matches:

```
checklist_eval/
├── llm_judge_v2.py
├── data/
│   ├── checklist.jsonl
│   └── your_agent_results.jsonl
└── evaluation_checklist_results/
```

## Notes

- The evaluator automatically skips files that have already been evaluated
- Results are written in real-time as evaluations complete
- Console output shows progress with ✓ (pass) and ✗ (fail) indicators
- All evaluations use the same judge model specified in `MODEL_NAME`

## Best Practices

1. **Start small**: Test with a few articles first to verify configuration
2. **Monitor API costs**: Track token usage, especially with large datasets
3. **Validate checklists**: Ensure checklist descriptions are clear and unambiguous
4. **Review results**: Manually verify a sample of evaluations for quality
5. **Adjust concurrency**: Balance speed with API rate limits and costs

