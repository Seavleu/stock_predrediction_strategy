from data_collection.data_preparation import fetch_stock_data, preprocess_data
from ai_models.lstm_model import LSTMPredictor
from sentiment.sentiment_analysis import fetch_sentiment_news
from strategies.trade_logic import generate_trade_signal

def run_trading_pipeline():
    """Execute the end-to-end stock trading system."""
    
    # Fetch real-time stock data
    df = fetch_stock_data('005930.KQ', source='naver')
    df, scaler = preprocess_data(df)
    
    # Run LSTM prediction
    model = LSTMPredictor()
    model.load_state_dict('best_model.pth')  # Load trained model
    predicted_price = model.predict(df)
    
    # Get sentiment score
    sentiment_score = fetch_sentiment_news('Samsung')
    
    # Generate trade signal
    trade_signal = generate_trade_signal(predicted_price, "Buy", sentiment_score)
    
    print(f"Predicted Price: {predicted_price:.2f}")
    print(f"Market Sentiment Score: {sentiment_score:.2f}")
    print(f"Generated Trade Signal: {trade_signal}")

if __name__ == "__main__":
    run_trading_pipeline()
