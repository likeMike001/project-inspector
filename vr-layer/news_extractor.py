# news_extractor.py
import io, csv
from typing import List, Dict
import requests

# Endpoint for your local synthetic-news generator service
NEWS_SERVICE_URL = "http://127.0.0.1:8000/generate_news_csv"


def generate_synthetic_news(topic: str, num_articles: int = 10) -> List[Dict]:
    """
    Ask the local LLM service to generate synthetic news for a topic,
    and return it as a list of dicts.

    Expected CSV columns from the service:
        id,topic,headline,body,source,published_at
    """
    payload = {"topic": topic, "num_articles": num_articles}

    resp = requests.post(NEWS_SERVICE_URL, json=payload, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    csv_text = data["csv"]  # the FastAPI service returns {"csv": "<csv_string>"}

    f = io.StringIO(csv_text)
    reader = csv.DictReader(f)

    articles: List[Dict] = []
    for row in reader:
        try:
            row["id"] = int(row["id"])
        except Exception:
            pass
        articles.append(row)

    return articles
