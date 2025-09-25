#
# Import Libraries
# os, csv, time: For file handling, timing, and environment.
# pandas: Easy table/data handling.
# rich.console, rich.table: For pretty printing in the terminal.
# openai: For model inference.
# langfuse: For logging and tracing all prompt/response runs.
#
import os
import csv
import time
import pandas as pd
from rich.console import Console
from rich.table import Table
from langfuse import Langfuse
from langfuse.openai import openai as openai_client
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

#
# Initialize Langfuse client
#
lf_client = Langfuse(
  secret_key=os.getenv("LF_SECRET_KEY"),  
  public_key=os.getenv("LF_PUBLIC_KEY"),  
  host="https://cloud.langfuse.com"
)

#
# Define Prompt Versions
# Labels for each style/version of prompt you want to test.
#
VERSION_1 = "v1_simple_instruction"
VERSION_2 = "v2_contextual_instruction"
VERSION_3 = "v3_detailed_prompt"

#
# Load Inputs (From CSV or Default List)
# Sample Inputs (either load from CSV or hard‐code as list)
# Loads your evaluation prompts either from a file or a hard-coded list.
# Each sample is a dictionary with an ID and a prompt text.
#
INPUTS_CSV = "data/inputs.csv"

if os.path.exists(INPUTS_CSV):
    samples = []
    with open(INPUTS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({"id": row["id"], "text": row["text"]})
else:
    # Fallback hard-coded list
    samples = [
        {"id": "1", "text": "Explain the benefits of renewable energy."},
        {"id": "2", "text": "Summarize the plot of 'The Lord of the Rings' in two sentences."},
        {"id": "3", "text": "Translate the following sentence to French: 'Artificial intelligence is transforming the world.'"},
        {"id": "4", "text": "List three best practices for secure API design."},
    ]

console = Console()

# 
# Define Prompt Templates
# build_prompt_v1: A simple instruction (short, direct).
# build_prompt_v2: Contextual, adds persona (“You are an expert assistant…”).
# build_prompt_v3: Very detailed, adds formatting guidelines and style.
# Each function builds a different version of the prompt for the same input.
# 
def build_prompt_v1(user_text: str) -> str:
    return f"Please answer the following question concisely:\n\n{user_text}"

def build_prompt_v2(user_text: str) -> str:
    return (
        "You are an expert assistant. Provide a helpful answer to the user's request below, "
        "being as informative as possible:\n\n"
        f"User: {user_text}\n\n"
        "Assistant:"
    )

def build_prompt_v3(user_text: str) -> str:
    return (
        "You are a professional technical writer. When responding:\n"
        "- Use bullet points when appropriate.\n"
        "- Keep language clear and precise.\n"
        "- Provide any relevant context.\n\n"
        f"Question: {user_text}\n\n"
        "Answer:"
    )

# 
# OpenAI + Langfuse Logging Wrapper
# Calls the OpenAI chat model with your prompt.
# Extracts the completion and token usage info.
# Logs prompt, response, metadata, version, and tokens to Langfuse for tracking/versioning.
# Returns both the model answer and token stats.
# 
def call_openai_with_lf(prompt: str, version_name: str) -> dict:
    # Call OpenAI ChatCompletion
    #model='gpt-4.1'
    #response = openai_client.chat.completions.create(
    #    model=model,
    #    messages=[{"role": "user", "content": prompt}],
    #)
    #choice = response.choices[0].message.content.strip()
    #usage = response.usage
    #lf_client.auth_check()
    choice = None
    usage = None
    # Log run to Langfuse: including prompt, response, and metadata
    try:
        response = openai_client.chat.completions.create(
            name="Compare prompts",                     # shows up as trace/generation name in Langfuse
            model="gpt-4.1",                 # keep your model; use any you like
            messages=[{"role": "user", "content": prompt}],
    # The following extras are parsed by Langfuse and attached to the trace:
            metadata={"provider": "openai", 
              "model": "gpt-4.1", 
              "langfuse_tags": ["streamlit", "prompt-engineering", "example"], 
              "langfuse_session_id":str(uuid.uuid4()), 
              "langfuse_user_id":str(uuid.uuid4()),
              "prompt_version": version_name
              }
        )
        #lf_client.trace(
        #    input=prompt,
        #    output=choice,
        #    tags=[version_name],
        #    version=version_name,
        #)
        choice = response.choices[0].message.content.strip()
        usage = response.usage
    except Exception as e:
        console.print(f"[red]Langfuse logging error:[/red] {e}")
    return {"output": choice, "tokens": usage}

# 
# Run All Prompt Versions for All Inputs
# For each sample and each prompt version:
# Builds the appropriate prompt.
# Calls the model and logs to Langfuse.
# Collects results (including token counts and output).
# Waits 1 second to avoid rate limits.
# 
results = []

for sample in samples:
    user_id = sample["id"]
    user_text = sample["text"]
    prompts = {
        VERSION_1: build_prompt_v1(user_text),
        VERSION_2: build_prompt_v2(user_text),
        VERSION_3: build_prompt_v3(user_text),
    }
    for version_name, prompt_text in prompts.items():
        console.print(f"[bold cyan]Running version:[/bold cyan] {version_name} on ID {user_id}")
        result = call_openai_with_lf(prompt_text, version_name)
        results.append(
            {
                "input_id": user_id,
                "version": version_name,
                "prompt": prompt_text,
                "output": result["output"],
                "tokens_prompt": result["tokens"].prompt_tokens,
                "tokens_completion": result["tokens"].completion_tokens,
                "tokens_total": result["tokens"].total_tokens,
            }
        )
        time.sleep(1.0)

# 
# Aggregate Results and Display
# Loads all results into a Pandas DataFrame.
# Prints a preview of results and a nicely formatted table showing prompt versioning results, tokens used, and truncated output.
# 
df = pd.DataFrame(results)

console.print("\n[bold green]Sample of logged results:[/bold green]")
console.print(df.head(), "\n")

table = Table(title="Prompt Versioning Comparison", show_lines=True)
cols = ["input_id", "version", "tokens_total", "tokens_prompt", "tokens_completion", "output"]
for col in cols:
    table.add_column(col)
for _, row in df.iterrows():
    table.add_row(
        str(row["input_id"]),
        row["version"],
        str(row["tokens_total"]),
        str(row["tokens_prompt"]),
        str(row["tokens_completion"]),
        row["output"].replace("\n", " ")[:80] + "…"
    )
console.print(table)

# 
# Save All Results
# Saves all results to a CSV file for further offline analysis or reporting.
# 
output_csv = "prompt_versioning_langfuse_results.csv"
df.to_csv(output_csv, index=False)
console.print(f"\n[bold yellow]All results saved to:[/bold yellow] {output_csv}")

#
# Summary: What does this script do?
# Tests multiple prompt versions on the same input.
# Logs each prompt/response pair, version, and token stats to Langfuse for full experiment traceability.
# Displays and saves all results for later review—so you can see which prompt version works best for your use case (in terms of quality or cost).
#