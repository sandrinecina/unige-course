# Prompt Engineering Sandbox with Streamlit, OpenAI, and Langfuse

This project provides a lightweight, interactive web UI for experimenting with prompt engineering using OpenAI's latest models and logs all interactions to Langfuse for analysis.

## Features

- Edit and test prompt templates live in a browser
- Use any available OpenAI model (default: GPT-4.1)
- See responses instantly and compare prompt versions
- Log all prompt/response pairs with metadata to Langfuse

## Quick Start

### 1. Clone and Install

```bash
pip install -r requirements.txt
```

### 2. Set Your Keys

Replace the placeholders in the script with your own:
- `OPENAI_API_KEY` from your [OpenAI dashboard](https://platform.openai.com/account/api-keys)
- `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` from your [Langfuse project settings](https://cloud.langfuse.com/)

### 3. Run the App

```bash
streamlit run app.py
```

### 4. Usage

- Enter your **prompt template** (e.g. "Summarize this text: {{input}}")
- Enter input text
- View and compare responses from OpenAI
- Check all runs in your [Langfuse dashboard](https://cloud.langfuse.com/)

## Example

```
Prompt Template: Summarize this text: {{input}}
Input Text: The quick brown fox jumps over the lazy dog.
```

## Requirements

- Python 3.8+
- See `requirements.txt` for details

