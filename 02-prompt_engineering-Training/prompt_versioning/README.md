# Prompt Versioning & Logging with OpenAI and Langfuse

This project demonstrates how to systematically compare different prompt versions using OpenAI's API, with automatic logging to Langfuse for prompt management, analytics, and experiment tracking.

## Features

- Easily define and compare multiple prompt versions
- Run on a batch of inputs (from CSV or in-code)
- Logs every prompt, output, and token usage to Langfuse
- Shows results in a pretty console table
- Saves all results to CSV for further analysis

## Quick Start

### 1. Install Requirements

```
pip install openai langfuse pandas rich
```

### 2. Set Your API Keys

Replace in the script:
- `OPENAI_API_KEY` with your [OpenAI API key](https://platform.openai.com/account/api-keys)
- `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` from your [Langfuse dashboard](https://cloud.langfuse.com/)

### 3. Prepare Inputs

- Either create a `data/inputs.csv` with columns `id,text`
- Or use the default in-code examples

### 4. Run the Script

```
python <your_script>.py
```

- Results will print to the console (with color tables) and save to `prompt_versioning_langfuse_results.csv`

## Example Prompt Versions

- **Simple Instruction:** "Please answer the following question concisely: ..."
- **Contextual Instruction:** "You are an expert assistant. Provide a helpful answer ..."
- **Detailed Prompt:** "You are a professional technical writer. When responding: ..."

## What is Langfuse?

[Langfuse](https://langfuse.com/) is an observability and prompt management platform for LLM applications. It logs all your prompts, responses, and versions for easy experiment tracking and analytics.

## Credits

- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Langfuse](https://langfuse.com/)
- [Rich Console](https://rich.readthedocs.io/)
- [Pandas](https://pandas.pydata.org/)

