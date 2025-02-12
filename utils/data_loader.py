import pandas as pd

def load_korean_stock_data(file_path):
    # Ensure the file is loaded with proper header and timestamp parsing
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    
    # Convert the 'timestamp' column to datetime (if it's not already) and set it as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    return df
