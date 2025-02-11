from utils.data_loader import load_korean_stock_data
from strategies.overlap_studies import BBANDS_indicator
from utils.trade_logic import execute_trade_logic

# Load the dataset
data = load_korean_stock_data('/path/to/korean_stock_data.csv')

# Run strategy on the data
signals = []
for i in range(len(data)):
    signal = execute_trade_logic(data.iloc[i:i+1], BBANDS_indicator)
    signals.append(signal)

# Output results
for signal in signals:
    print(signal)
