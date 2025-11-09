# fetch_news.py
from news_extractor import generate_synthetic_news


def main():
    topic = "Ethereum"
    num_articles = 10

    articles = generate_synthetic_news(topic, num_articles)

    for article in articles:
        print(f"ID:           {article['id']}")
        print(f"Topic:        {article['topic']}")
        print(f"Headline:     {article['headline']}")
        print(f"Source:       {article['source']}")
        print(f"Published At: {article['published_at']}")
        print("Body (first 300 chars):")
        print(article['body'][:300])
        print("-" * 80)


if __name__ == "__main__":
    main()
