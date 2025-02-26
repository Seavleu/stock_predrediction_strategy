import pandas as pd

def load_stock_data(filepath):
    """Load stock data and standardize column names."""
    df = pd.read_csv(filepath, parse_dates=["Date" if "Date" in pd.read_csv(filepath, nrows=1).columns else "timestamp"])

    # Standardize column names
    if "Date" in df.columns:  # KOSPI dataset
        df.rename(columns={
            "Date": "date",
            "Open": "opening_price",
            "High": "highest_price",
            "Low": "lowest_price",
            "Close": "closing_price",
            "Volume": "trading_volume"
        }, inplace=True)

    return df
