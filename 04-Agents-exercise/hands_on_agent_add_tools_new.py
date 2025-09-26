# -*- coding: utf-8 -*-
"""
Multi-tool agent using LlamaIndex (v0.10+), OpenAI, LangChain Community, and Langfuse.
- Tools: Calculator, OpenAI chat, Wikipedia
- Tracing: Langfuse @observe decorator + OpenAI native integration
"""

import os
import sys
import asyncio

# --- Langfuse
from langfuse import get_client, observe
from langfuse.openai import openai  # auto-traces OpenAI calls

# --- OpenAI client (uses OPENAI_API_KEY from env)
openai_client = openai.OpenAI()

# --- LlamaIndex
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI

# --- LangChain Community Wikipedia tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# ======================
# Setup: API Keys
# ======================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # required by OpenAI + LlamaIndex wrapper

# Initialize Langfuse client (reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST)
langfuse = get_client()

# ======================
# Tool: Wikipedia
# ======================
@observe(name="wikipedia", as_type=None)
def wiki_tool(query: str) -> str:
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wikipedia.run(query)

wiki_func_tool = FunctionTool.from_defaults(
    fn=wiki_tool,
    name="Wikipedia",
    description="Search Wikipedia for a summary."
)

# ======================
# Tool: Calculator
# ======================
@observe(name="calculator", as_type=None)
def calculator(expr: str) -> str:
    try:
        result = str(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        result = "Invalid math expression."
    return result

calc_tool = FunctionTool.from_defaults(
    fn=calculator,
    name="Calculator",
    description="Evaluate a simple math expression (e.g., '2+2*3')."
)

# ======================
# Tool: OpenAI Chat
# ======================
@observe(name="openai-chat", as_type="generation")
def openai_chat(prompt: str) -> str:
    """
    Uses OpenAI Chat Completions API (gpt-4.1 family).
    Langfuse traces automatically via langfuse.openai.
    """
    resp = openai_client.chat.completions.create(
        model="gpt-4.1",  # or "gpt-4.1-mini"
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()

openai_tool = FunctionTool.from_defaults(
    fn=openai_chat,
    name="OpenAIChat",
    description="Ask a general knowledge question via OpenAI."
)

#
# Weather in city Tool
# get_weather(): Fetches current weather for a given city using Open-Meteo API.
# Logs the input/output to Langfuse for analytics.
#
@observe(name="Get Weather", as_type=None)
def get_weather(city: str) -> str:
    import requests
    try:
        response = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}")
        data = response.json()
        lat = data["results"][0]["latitude"]
        lon = data["results"][0]["longitude"]
        weather_response = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        )
        weather_data = weather_response.json()
        temp = weather_data["current_weather"]["temperature"]
        wind = weather_data["current_weather"]["windspeed"]
        result = f"The current temperature in {city} is {temp}°C with windspeed {wind} km/h."
        return result
    except Exception as e:
        return f"Could not fetch weather data: {e}"

weather_func_tool = FunctionTool.from_defaults(
    fn=get_weather, name="Weather", description="Geat current weather for a city."
)


#
# News Tool
# get_weather(): Fetches recent news from NEWS server
# Logs the input/output to Langfuse for analytics.
#
@observe(name="Get News", as_type=None)
def news_tool(topic: str, language: str = "en") -> str:
    try:
        url = "http://localhost:8000/api"
        payload = {"topic": topic, "language": language}
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        return response.json().get("result", "No result returned.")
    except Exception as e:
        return f"Error calling News server: {e}"

fun_news_tool = FunctionTool.from_defaults(
    fn=news_tool,
    name="News",
    description="Gets up-to-date news headlines about a topic."
)

# ======================
# Agent Setup (LlamaIndex)
# ======================
TOOLS = [calc_tool, openai_tool, wiki_func_tool, weather_func_tool]
llm = LlamaIndexOpenAI(model="gpt-4.1", api_key=OPENAI_API_KEY)
agent = FunctionAgent(
    tools=TOOLS,
    llm=llm,
    system_prompt="You are a helpful agent. Use the available tools if helpful."
)

# ======================
# Async CLI
# ======================
async def main():
    # On Windows, ensure a selector loop policy
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    print("LlamaIndex OpenAI Agent (exit with 'exit' or 'quit')")
    loop = asyncio.get_running_loop()

    while True:
        # non-blocking input in async code
        query = await loop.run_in_executor(None, input, "\n> ")
        if query.strip().lower() in ("exit", "quit"):
            break

        # Group each turn under a Langfuse span (now there IS a running loop)
        with langfuse.start_as_current_span(name="user-turn", metadata={"source": "cli"}):
            res = await agent.run(user_msg=query)
            print("\n" + str(res))
            langfuse.update_current_trace(tags=["cli", "demo"])

        # Flush Langfuse buffers
        langfuse.flush()
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(main())