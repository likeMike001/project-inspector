from fastapi import FastAPI
from pydantic import BaseModel
import csv, io
from datetime import datetime, timedelta
import random
import os
from dotenv import load_dotenv
from anthropic import Anthropic


load_dotenv()
app = FastAPI()
client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))


class NewsRequest(BaseModel):
    topic: str
    num_articles: int = 10



@app.post("/generate_news_csv")
def generate_news_csv(req: NewsRequest):
    """
    Use Claude to generate synthetic financial news in CSV format.
    """
    prompt = f"""
    Generate {req.num_articles} short financial news articles about {req.topic}.
    Each article should look like something from Bloomberg or Reuters.

    Output them **only** as CSV with these columns:
    id,topic,headline,body,source,published_at

    Rules:
    - id starts from 1
    - topic = {req.topic}
    - source = "synthetic_claude"
    - published_at = current UTC timestamp in ISO 8601 format (Z at end)
    - body = 2–3 sentences of coherent financial context
    - no markdown, no commentary, only CSV text.
    """

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )

    csv_text = response.content[0].text.strip()
    return {"csv": csv_text}