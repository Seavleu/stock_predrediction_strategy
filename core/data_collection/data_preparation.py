import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler 

def prepare_data(df, target_col='close', lookback_days=60, train_split=0.8, val_split=0.1):
    """Prepare stock data for LSTM training."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    data = df[target_col].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(lookback_days, len(scaled_data)):
        X.append(scaled_data[i - lookback_days:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)

    train_size = int(len(X) * train_split)
    val_size = int(len(X) * val_split)

    return X[:train_size], y[:train_size], X[train_size:train_size+val_size], y[train_size:train_size+val_size], X[train_size+val_size:], y[train_size+val_size:], scaler

def rank_top_5_stocks(df):
    """Rank top 5 companies based on average price change & volume."""
    df["daily_return"] = df["closing_price"].pct_change()
    stock_performance = df.groupby("company").agg(
        avg_return=("daily_return", "mean"),
        avg_volume=("trading_volume", "mean")
    )

    # Rank by avg return and volume, select top 5
    stock_performance["score"] = stock_performance["avg_return"] + stock_performance["avg_volume"]
    top_5_stocks = stock_performance.sort_values(by="score", ascending=False).head(5).index.tolist()

    return df[df["company"].isin(top_5_stocks)]


# def fetch_stock_data(ticker, source='yfinance'):
#     """Fetch stock data from various sources."""
#     if source == 'yfinance':
#         url = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}'
#     elif source == 'naver':
#         url = f'https://finance.naver.com/api/stocks/{ticker}'
#     elif source == 'kis':
#         url = f'https://api.kis.com/stock/{ticker}'
    
#     response = requests.get(url)
#     data = pd.read_csv(response.text) if source == 'yfinance' else pd.DataFrame(response.json())
    
#     return data

# def preprocess_data(df):
#     """Normalize and preprocess stock data for model input."""
#     scaler = MinMaxScaler()
#     df['scaled_close'] = scaler.fit_transform(df[['Close']])
    
#     return df, scaler
