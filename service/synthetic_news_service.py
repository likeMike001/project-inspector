from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from anthropic import Anthropic
from datetime import datetime
from pathlib import Path

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
    Also saves the CSV into the ../data folder.
    """
    prompt = f"""
You are a CSV generator. Respond with ONLY valid CSV, no markdown, no code fences.

Generate {req.num_articles} short financial news articles about {req.topic}.
Each article should look like something from Bloomberg or Reuters.

Output format:
- First row MUST be the exact header:
  id,topic,headline,body,source,published_at
- Each subsequent row MUST be a CSV row with:
  id (integer, starting from 1),
  topic (exactly {req.topic}),
  headline,
  body,
  source (exactly "synthetic_claude"),
  published_at (ISO 8601 timestamp with Z, between 2025-01-14T00:00:00Z and 2025-11-09T00:00:00Z).

Additional rules:
- Spread the published_at values across that date range (do NOT use the same timestamp for every row).
- If a field contains a comma or double quote, wrap the field in double quotes and escape inner quotes by doubling them, like proper CSV.
- Do NOT wrap the CSV in ``` backticks or markdown.
- Do NOT include any commentary or explanation, only the raw CSV.
    """

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )

    csv_text = response.content[0].text.strip()

    # Just in case Claude still wraps something
    if csv_text.startswith("```"):
        csv_text = csv_text.strip("`").strip()
        if csv_text.lower().startswith("csv"):
            csv_text = csv_text[3:].lstrip()

    # 🔹 Save to ../data/ with timestamped filename
    base_dir = Path(__file__).resolve().parent  # .../project-one/service
    data_dir = base_dir.parent / "data"         # .../project-one/data
    data_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_topic = req.topic.replace(" ", "_")
    file_path = data_dir / f"synthetic_news_{safe_topic}_{timestamp}.csv"

    with file_path.open("w", encoding="utf-8") as f:
        f.write(csv_text)

    print(f"✅ Saved synthetic news to {file_path}")

    return {"csv": csv_text, "file_path": str(file_path)}
