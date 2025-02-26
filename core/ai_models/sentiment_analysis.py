import requests
from textblob import TextBlob

def fetch_sentiment_news(company):
    """Fetch news articles and perform sentiment analysis."""
    url = f'https://newsapi.org/v2/everything?q={company}&apiKey={MY_API_KEY}'
    response = requests.get(url).json()
    
    articles = [article['title'] for article in response['articles']]
    sentiment_scores = [TextBlob(article).sentiment.polarity for article in articles]
    
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    return avg_sentiment
