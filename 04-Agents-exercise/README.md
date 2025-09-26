# LlamaIndex Tool-Using Agent: OpenAI, Wikipedia, Calculator + Langfuse Logging

This project demonstrates a modern agent architecture using LlamaIndex, OpenAI’s GPT-4, LangChain Community tools, and Langfuse for complete prompt and tool use tracking.  
You can ask questions, do math, or search Wikipedia—all from a simple CLI!

---

## Features

- **LlamaIndex Agent**: Uses modern function-calling interface.
- **OpenAI Integration**: Access GPT-4.1 for general knowledge queries.
- **Wikipedia Tool**: Retrieve summaries directly from Wikipedia.
- **Calculator Tool**: Perform basic math in natural language.
- **Langfuse Logging**: All tool/model calls (input/output, tokens, tags) are logged to Langfuse for analysis and prompt/version management.
- **Easy to Extend**: Add your own tools with just a few lines.

---

## Quick Start

### 1. Install Requirements

```bash
pip install openai llama-index langfuse langchain-community
```

### 2. Configure Your API Keys

Edit the script and replace:
- `OPENAI_API_KEY` with your [OpenAI API Key](https://platform.openai.com/account/api-keys)
- `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` from your [Langfuse dashboard](https://cloud.langfuse.com/)

### 3. Run the Agent

```bash
python agent.py
```

You will see a prompt:
```
LlamaIndex OpenAI Agent (exit with 'exit' or 'quit')

>
```
Ask any question, math query, or Wikipedia topic!

---

## Example Usage

```
> What is the capital of Germany?
Berlin.

> 12 * 19 + 8
236

> Who is Ada Lovelace? (Wikipedia)
Ada Lovelace was an English mathematician and writer, chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical Engine...
```

All tool uses and model calls are automatically logged in your Langfuse dashboard for review and analytics.

---

## Components Used

- [LlamaIndex](https://github.com/run-llama/llama_index)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Langfuse](https://langfuse.com/) (for prompt/tool logging)
- [LangChain Community Tools](https://python.langchain.com/docs/community) (Wikipedia)
- [WikipediaAPIWrapper](https://github.com/langchain-ai/langchain)

---

## Customization

- Add more tools by defining a new function and wrapping with `FunctionTool.from_defaults`.
- Swap to different LLMs supported by LlamaIndex.
- Modify the system prompt to tune the agent’s style.
