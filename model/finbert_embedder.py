# finbert_embedder.py
import ast
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification



DATA_DIR = Path("../data")
ETH_CSV_PATH = Path("merged_eth_staking_dynamic.csv")
NEWS_CSV_PATH = Path("../data/synthetic_news_string_20251109_060057.csv")
OUTPUT_CSV_PATH = DATA_DIR / "final_with_sentiment.csv"


def get_latest_news_csv(data_dir: Path) -> Path:
    """
    Find the most recent synthetic_news_*.csv file in ../data.
    If you prefer, you can hardcode a path instead.
    """
    candidates = sorted(data_dir.glob("synthetic_news_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No synthetic_news_*.csv found in {data_dir}")
    return candidates[-1]  # newest by name (timestamp in filename)


def clean_synthetic_news(path: str | Path) -> pd.DataFrame:
    """
    Load the synthetic news CSV produced by the Claude service.

    Assumes the file looks like:

        id,topic,headline,body,source,published_at
        1,Ethereum,"Headline...","Body...","synthetic_claude",2025-03-01T10:15:00Z
        ...

    We just need to:
      - read it as normal CSV
      - parse published_at as datetime
      - drop any rows with invalid timestamps
    """
    df = pd.read_csv(path)

    required_cols = {"id", "topic", "headline", "body", "source", "published_at"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"News CSV missing columns: {missing}")

    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")

    print("Raw news rows:", len(df))
    print("Null published_at:", df["published_at"].isna().sum())

    df = df.dropna(subset=["published_at"]).reset_index(drop=True)

    print("After dropna, news rows:", len(df))
    return df


def apply_finbert_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run FinBERT (ProsusAI/finbert) over the 'body' column and
    add sentiment columns: sentiment_label, sent_pos, sent_neg, sent_neu.
    """
    print("Loading FinBERT...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()

    texts = df["body"].fillna("").tolist()

    # Batch everything at once (you only have ~10 articles)
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)

    # FinBERT class order is [positive, negative, neutral]
    labels = ["positive", "negative", "neutral"]

    df["sentiment_label"] = [labels[i] for i in probs.argmax(dim=-1)]
    df["sent_pos"] = probs[:, 0].numpy()
    df["sent_neg"] = probs[:, 1].numpy()
    df["sent_neu"] = probs[:, 2].numpy()

    return df


def merge_with_eth_data(news_df: pd.DataFrame, eth_path: str | Path) -> pd.DataFrame:
    """
    Merge the ETH metrics with the FinBERT-enriched news by timestamp.

    We take the latest news at or before each ETH 'timestamp' (backward join),
    with a moderate tolerance window so news only influences nearby rows.
    """
    eth = pd.read_csv(eth_path)
    eth["timestamp"] = pd.to_datetime(eth["timestamp"], utc=True, errors="coerce")

    # Drop any ETH rows without a valid timestamp
    eth = eth.dropna(subset=["timestamp"]).reset_index(drop=True)

    # Ensure news published_at is datetime and has no nulls
    news_df["published_at"] = pd.to_datetime(
        news_df["published_at"], utc=True, errors="coerce"
    )
    print("News null published_at before drop:", news_df["published_at"].isna().sum())
    news_df = news_df.dropna(subset=["published_at"]).reset_index(drop=True)
    print("News rows after dropna:", len(news_df))

    # Sort both by time
    eth_sorted = eth.sort_values("timestamp")
    news_sorted = news_df.sort_values("published_at")

    print(
        "ETH time range:",
        eth_sorted["timestamp"].min(),
        "→",
        eth_sorted["timestamp"].max(),
    )
    print(
        "News time range:",
        news_sorted["published_at"].min(),
        "→",
        news_sorted["published_at"].max(),
    )

    # Use a 1-day tolerance so each article only applies to nearby ETH rows
    merged = pd.merge_asof(
        eth_sorted,
        news_sorted,
        left_on="timestamp",
        right_on="published_at",
        direction="backward",
        tolerance=pd.Timedelta(days=2),
    )

    return merged


if __name__ == "__main__":
    print("🔹 Cleaning synthetic news...")
    news = clean_synthetic_news(NEWS_CSV_PATH)

    print("🔹 Applying FinBERT sentiment...")
    news = apply_finbert_sentiment(news)

    print("🔹 Merging with ETH dataset...")
    merged = merge_with_eth_data(news, ETH_CSV_PATH)

    # 🔹 Post-processing: handle missing news / sentiment
    # 1 = row has some news attached, 0 = no news
    merged["has_news"] = merged["headline"].notna().astype(int)

    # If there was no nearby news, sentiment columns will be NaN.
    # Treat that as "no signal" → neutral baseline.
    for col, default in [
        ("sent_pos", 0.0),
        ("sent_neg", 0.0),
        ("sent_neu", 1.0),
    ]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(default)

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"✅ Final dataset saved to {OUTPUT_CSV_PATH}")