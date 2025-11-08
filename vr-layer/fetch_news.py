from gnews import GNews

google_news = GNews(language='en', country='US', period='7d', max_results=10)
results = google_news.get_news('Ethereum')


for news in results:
    print(f"Title: {news['title']}")
    print(f"Description: {news['description']}")
    print(f"URL: {news['url']}")
    print(f"Published At: {news['published date']}")
    # print(f"Content: {news['content']}")
    print("-" * 80)
    
    
