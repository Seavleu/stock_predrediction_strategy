# Training
######################################################################################
from strategies.overlap_studies import BBANDS_indicator, EMA_indicator
from strategies.momentum_indicators import MACD_indicator, RSI_indicator, STOCH_indicator, WILLR_indicator, MFI_indicator
from strategies.volume_indicators import CMF_indicator, AD_indicator

from utils.data_loader import load_stock_data
from utils.trade_logic import execute_trade_logic 

DATA_PATH = "data/kospi/kospi_weekly_7y.csv"
data = load_stock_data(DATA_PATH)

# List to store the signals and trades
signals = []
trades = []

portfolio_value = 10000  # Starting portfolio value
position = 0  # No position initially
cash = portfolio_value

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
    signal_stoch = STOCH_indicator(data.iloc[i:i+1])
    signal_ad = AD_indicator(data.iloc[i:i+1])
    
    # Append all signals to the list
    signals.append({
        "Date": data.iloc[i]["date"].strftime('%Y-%m-%d'), 
        "BBANDS": signal_bbands,
        "EMA": signal_ema,
        "MACD": signal_macd,
        "RSI": signal_rsi,
        "MFI": signal_mfi,
        "WILLR": signal_willr,
        "CMF": signal_cmf,
        "STOCH": signal_stoch,
        "AD": signal_ad
    })

    # Aggregate signals (use majority rule or any other aggregation strategy)
    buy_signals = sum(1 for signal in signals[-1].values() if signal == "Buy")
    sell_signals = sum(1 for signal in signals[-1].values() if signal == "Sell")

    # Decide on trade based on majority rule
    if buy_signals > sell_signals:
        trade_signal = "Buy"
    elif sell_signals > buy_signals:
        trade_signal = "Sell"
    else:
        trade_signal = "Hold"

    # Execute trade and track portfolio
    if trade_signal == "Buy" and position == 0:  # Buy if no position
        position = cash / data.iloc[i]["closing_price"]
        cash = 0
        trades.append(f"Buy at {data.index[i]} - {data.iloc[i]['closing_price']}")
        
    elif trade_signal == "Sell" and position > 0:  # Sell if holding position
        cash = position * data.iloc[i]["closing_price"]
        position = 0
        trades.append(f"Sell at {data.index[i]} - {data.iloc[i]['closing_price']}")

    # Print portfolio status after each transaction
    current_value = cash + (position * data.iloc[i]["closing_price"]) if position > 0 else cash
    print(f"Date: {data.iloc[i]['date'].strftime('%Y-%m-%d')}, Portfolio Value: {current_value}")  

# Final portfolio value (cash + any position left)
final_value = cash + (position * data.iloc[-1]["closing_price"]) if position > 0 else cash
print(f"\nFinal Portfolio Value: {final_value}")
print("Trades:", trades)

print("\nAll Signals:")
for signal in signals:
    print(signal)