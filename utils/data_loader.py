import pandas as pd

def load_stock_data(filepath):
    """Load stock data and ensure flexible column naming."""
    df = pd.read_csv(filepath)

    if {"opening_price", "highest_price", "lowest_price", "closing_price", "trading_volume"}.issubset(df.columns):
        df.rename(columns={
            "opening_price": "Open",
            "highest_price": "High",
            "lowest_price": "Low",
            "closing_price": "Close",
            "trading_volume": "Volume"
        }, inplace=True)

    if "timestamp" in df.columns:
        df.rename(columns={"timestamp": "Date"}, inplace=True)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    return df
