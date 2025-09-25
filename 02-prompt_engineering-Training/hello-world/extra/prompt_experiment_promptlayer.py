#
# streamlit: Used to build a simple, interactive web UI.
# promptlayer: Provides an API for logging/tracking prompt engineering experiments and calls to LLMs (in this case, OpenAI).
# os: Used to set environment variables for your API keys.
#
import streamlit as st
from promptlayer import PromptLayer
from openai import OpenAI
import os

#
# Sets the OpenAI API key in the environment so any OpenAI call can find it.
# Initializes a PromptLayer client with your PromptLayer API key.
# Accesses the OpenAI API through PromptLayer’s wrapper (so all requests are tracked in PromptLayer).
# Creates an OpenAI client via PromptLayer’s wrapper.
#
openai_client = OpenAI()
promptlayer_client = PromptLayer(api_key=os.getenv("PROMPT_LAYER_KEY"), enable_tracing=True)

# #
# Streamlit UI:
# st.title: Adds a big page title.
# st.text_area: Lets the user enter or modify a prompt template (default: "Summarize this text: {{input}}").
# st.text_input: Lets the user enter any text to be summarized (or inserted into the prompt).
# #
st.title("Prompt Engineering Sandbox")
base_prompt = st.text_area("Prompt Template", "Summarize this text: {{input}}")
user_input = st.text_input("Input Text")

# #
# if user_input:: Only runs when the user has entered something.
# Prompt Filling: Replaces {{input}} in the prompt template with what the user typed.
# Call OpenAI via PromptLayer:
# Uses the PromptLayer-wrapped OpenAI client to call the gpt-4.1 chat model with the user’s prompt.
# The call is logged to PromptLayer for experiment tracking and prompt versioning.
# Sets max_tokens=100 to limit the response length.
# Display Result: Shows the model’s response (from the first choice/message) in the Streamlit app.
# #
if user_input:
    full_prompt = base_prompt.replace("{{input}}", user_input)
    response = openai_client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": full_prompt}],
        max_tokens=100
    )
      # Debugging line to see the response structure
    st.write(response.choices[0].message.content)
    response = promptlayer_client.run(
    prompt_name="test",
    input_variables={
        "name": full_prompt
    },
    metadata={
        "user_id": "12345"
    }
)
    


# What does this app do?
#
# Lets you rapidly experiment with prompt engineering (try different prompt templates and inputs).
# All prompts, inputs, and results are tracked in PromptLayer, so you can analyze history and compare prompt versions over time.
# Uses a simple, user-friendly UI thanks to Streamlit.
# What can you modify or extend?
#
# Try different OpenAI models ("gpt-3.5-turbo", "gpt-4", etc.).
# Add sliders or dropdowns for model selection, temperature, or max_tokens.
# Track multiple prompts and compare outputs in PromptLayer dashboard.
# Add history or response side-by-side comparison in the UI.