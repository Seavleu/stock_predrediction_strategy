import pandas as pd
from strategies.overlap_studies import BBANDS_indicator

def backtest_strategy(data, strategy):
    """Test the strategy on historical data."""
    results = []
    for i in range(len(data)):
        result = strategy(data.iloc[:i+1])  # Apply the strategy on historical data
        results.append(result)
    return results
