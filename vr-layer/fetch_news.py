# fetch_news.py
from news_extractor import generate_synthetic_news
import pandas as pd 
from datetime import datetime

def main():
    topic = "Ethereum"
    num_articles = 10

    articles = generate_synthetic_news(topic, num_articles)

    df = pd.DataFrame(articles)
    
    time_stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    file_path = f"../data/synthetic_news_{topic}_{time_stamp}.csv"
    
    df.to_csv(file_path, index=False)
    print(f"Synthetic news articles saved to {file_path}")
   
   
if __name__ == "__main__":
    main()
