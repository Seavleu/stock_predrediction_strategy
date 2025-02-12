from strategies.overlap_studies import BBANDS_indicator, EMA_indicator
from strategies.momentum_indicators import MACD_indicator, RSI_indicator, MFI_indicator, WILLR_indicator, CMF_indicator, AROON_indicator, STOCH_indicator, MACDEXT_indicator, AD_indicator
from utils.data_loader import load_korean_stock_data
from utils.trade_logic import execute_trade_logic

# Load dataset
data = load_korean_stock_data('data/korean_stock_data.csv')

# List to store the signals
signals = []

# Loop through each data point
for i in range(len(data)):
    # Existing indicators
    signal_bbands = BBANDS_indicator(data.iloc[i:i+1])
    signal_ema = EMA_indicator(data.iloc[i:i+1])
    signal_macd = MACD_indicator(data.iloc[i:i+1])
    signal_rsi = RSI_indicator(data.iloc[i:i+1])

    # New indicators
    signal_mfi = MFI_indicator(data.iloc[i:i+1])
    signal_willr = WILLR_indicator(data.iloc[i:i+1])
    signal_cmf = CMF_indicator(data.iloc[i:i+1])
    signal_aroon = AROON_indicator(data.iloc[i:i+1])
    signal_stoch = STOCH_indicator(data.iloc[i:i+1])
    signal_macd_ext = MACDEXT_indicator(data.iloc[i:i+1])
    signal_ad = AD_indicator(data.iloc[i:i+1])
    
    # Append all signals to the list
    signals.append({
        "Date": data.index[i],
        "BBANDS": signal_bbands,
        "EMA": signal_ema,
        "MACD": signal_macd,
        "RSI": signal_rsi,
        "MFI": signal_mfi,
        "WILLR": signal_willr,
        "CMF": signal_cmf,
        "AROON": signal_aroon,
        "STOCH": signal_stoch,
        "MACDEXT": signal_macd_ext,
        "AD": signal_ad
    })

    # Execute the trading logic based on the strategy signals
    # Choose which strategy signal to base the decision on, here using BBANDS as an example
    trade_signal = signal_bbands  # You can change this to use any indicator's signal
    execute_trade_logic(data.iloc[i], trade_signal)

# Print all signals
for signal in signals:
    print(signal)


# {'Date': '2025-02-11', 'BBANDS': 'Hold', 'EMA': 'Buy', 'MACD': 'Buy', 'RSI': 'Neutral'}
# {'Date': '2025-02-10', 'BBANDS': 'Buy', 'EMA': 'Sell', 'MACD': 'Hold', 'RSI': 'Sell'}

