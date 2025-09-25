#
# Import Libraries
# streamlit: For building the interactive web UI.
# os: To set environment variables (like your API key).
# OpenAI: Official OpenAI Python SDK (v1.x).
# Langfuse: For logging prompt/response pairs (tracing) for observability and analysis.
# 
import streamlit as st
import os
from langfuse import Langfuse
from langfuse.openai import openai
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize Langfuse client
#
lf_client = Langfuse(
  secret_key=os.getenv("LF_SECRET_KEY"),  
  public_key=os.getenv("LF_PUBLIC_KEY"), 
  host="https://cloud.langfuse.com"
)

#
# Streamlit UI: Title and Prompt Setup
# st.title(...): Adds a page title at the top of the app.
# st.text_area(...): Lets the user define a prompt template.
# Default: "Summarize this text: {{input}}" (but user can edit!).
# st.text_input(...): Lets the user input the text that will be plugged into the prompt template.
#
st.title("Prompt Engineering Sandbox")

# Initialize session ID in Streamlit session state if it doesn't exist
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
# Can do the same with userID if needed

base_prompt = st.text_area("Prompt Template", "Summarize this text: {{input}}")
user_input = st.text_input("Input Text")

# 
# When the user enters input, this block:
# Replaces {{input}} in the prompt template with the actual user input.
# Calls OpenAI: Sends the fully constructed prompt to the gpt-4.1 model using the chat endpoint.
# The prompt is provided as a user message.
# You can swap 'gpt-4.1' for any available model (e.g., 'gpt-3.5-turbo').
# Extracts the output from the model’s response.
# Logs everything to Langfuse:
# Stores the prompt (input), the model’s answer (output), and a metadata dictionary (provider/model).
# Displays the answer in the Streamlit app with st.write(...).
# 
if user_input:
  full_prompt = base_prompt.replace("{{input}}", user_input)
  response = openai.chat.completions.create(
    name="chat",                     # shows up as trace/generation name in Langfuse
    model="gpt-4.1",                 # keep your model; use any you like
    messages=[{"role": "user", "content": full_prompt}],
    # The following extras are parsed by Langfuse and attached to the trace:
    metadata={"provider": "openai", 
              "model": "gpt-4.1", 
              "langfuse_tags": ["streamlit", "prompt-engineering", "example"], 
              "langfuse_session_id": st.session_state.session_id, 
              "langfuse_user_id":str(uuid.uuid4())},
    )
  output = response.choices[0].message.content
  st.write(output)


# 
# What does this let you do?
# Experiment with prompt templates and instantly see the effect of prompt engineering.
# Log every run (prompt + answer) for later review or prompt management via Langfuse.
# Use OpenAI’s latest models from a simple, interactive UI—great for teaching or prototyping.
# 
#
# How does it work for hands-on exercises?
# Learners can try different prompt structures, see the answers change, and review the history in Langfuse.
# Change the prompt template to anything, e.g.
# "Translate this to French: {{input}}"
# "Write a rhyming poem about: {{input}}"
# See the impact of small prompt tweaks right away.
# All data is saved (prompts/outputs) for further analysis or feedback loops.
# 





