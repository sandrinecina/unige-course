# fastapi: The web framework. Fast, modern, async, and easy to use.
# pydantic.BaseModel: Used to define the expected structure of input data (the request body).
# requests: To make HTTP requests to the NewsAPI.
# os: For reading environment variables (useful for API keys).
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

#This creates the web application object, to which you’ll add endpoints.
app = FastAPI()

# Go to https://newsapi.org/register and sign up for a free API key.
# Tries to read your NewsAPI key from the environment variable NEWSAPI_KEY.
# Falls back to "YOUR_NEWSAPI_KEY_HERE" if not set (be sure to replace this if you don’t use the environment variable).
# Tip: It’s best practice to keep secrets and keys out of your codebase and use environment variables.
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")  # Replace with your NewsAPI key or use an environment variable

# #Defines the expected format for incoming POST requests:
# topic: The news topic (required).
# language: The news language (optional, default "en" for English).
# This ensures automatic validation of incoming data and provides autocomplete/documentation in FastAPI’s docs.
class NewsRequest(BaseModel):
    topic: str  # e.g., "AI", "sports", "technology"
    language: str = "en"  # Optional: default to English

# Receives a POST request with a JSON payload (e.g., {"topic": "AI", "language": "en"}).
# Constructs parameters for the NewsAPI:
# "q": Query/topic
# "language": Language of articles
# "sortBy": Newest first
# "apiKey": Your NewsAPI key
# "pageSize": Limit to 5 headlines for clarity (can change as needed)
# Sends an HTTP GET request to NewsAPI's /everything endpoint.
# If successful (HTTP 200):
# Parses the JSON, extracts the list of articles, and creates a list of {"title", "url"}.
# If there are headlines, returns them as a formatted string (markdown-like list).
# If not, returns a friendly “no headlines” message.
# If error (non-200):
# Returns the error message from NewsAPI (helpful for debugging).
# All responses are standardized as {"result": ...}—easy for agents or clients to consume.
@app.post("/api")
async def get_news(req: NewsRequest):
    params = {
        "q": req.topic,
        "language": req.language,
        "sortBy": "publishedAt",
        "apiKey": NEWSAPI_KEY,
        "pageSize": 5
    }
    url = "https://newsapi.org/v2/everything"
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json()
        headlines = [
            {"title": a["title"], "url": a["url"]}
            for a in data.get("articles", [])
        ]
        if not headlines:
            return {"result": f"No headlines found for '{req.topic}'."}
        result = "\n".join([f"- {h['title']} ({h['url']})" for h in headlines])
    else:
        result = f"News API error: {r.json().get('message', 'Unknown error')}"
    return {"result": result}

#A simple GET endpoint for testing if your server is running (can open in browser or use curl).
@app.get("/")
async def root():
    return {"message": "Fast API News server is running!"}


# uvicorn main:app --reload --port 8000: Starts the FastAPI server on port 8000.
# --reload: Auto-reloads on code change (great for development).
# curl -X POST "http://localhost:8000/api" \
#      -H "Content-Type: application/json" \
#      -d '{"topic": "AI", "language": "en"}'