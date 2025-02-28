from strategies.momentum_indicators import (
    MACD_indicator, RSI_indicator, STOCH_indicator, WILLR_indicator, MFI_indicator,
    ADX_indicator, ATR_indicator, CCI_indicator, Parabolic_SAR_indicator, OBV_indicator,
    BBANDS_indicator, EMA_indicator
) 
from utils.data_loader import load_stock_data
from utils.trade_logic import execute_trade_logic_based_on_signals, execute_trade_logic

DATA_PATH = "data/kospi/kospi_daily_10y.csv"
data = load_stock_data(DATA_PATH)

portfolio_value = 10000  # Starting capital
position = 0  # No open position initially
cash = portfolio_value
signals = []
trades = []

# ✅ Iterate Over Data & Apply Trading Logic
for i in range(len(data)):
    row = data.iloc[i:i+1]

    # ✅ Compute Technical Indicators
    signal_bbands = BBANDS_indicator(row)
    signal_ema = EMA_indicator(row)
    signal_macd = MACD_indicator(row)
    signal_rsi = RSI_indicator(row)
    signal_mfi = MFI_indicator(row)
    signal_willr = WILLR_indicator(row)
    signal_stoch = STOCH_indicator(row)
    signal_adx = ADX_indicator(row)
    signal_atr = ATR_indicator(row)
    signal_cci = CCI_indicator(row)
    signal_sar = Parabolic_SAR_indicator(row)
    signal_obv = OBV_indicator(row)  

    # ✅ Store Signals
    signals.append({
        "Date": data.iloc[i]["Date"],
        "BBANDS": signal_bbands,
        "EMA": signal_ema,
        "MACD": signal_macd,
        "RSI": signal_rsi,
        "MFI": signal_mfi,
        "WILLR": signal_willr,
        "STOCH": signal_stoch,
        "ADX": signal_adx,
        "ATR": signal_atr,
        "CCI": signal_cci,
        "SAR": signal_sar,
        "OBV": signal_obv,
    })

    # ✅ Determine Trade Signal
    trade_signal = execute_trade_logic_based_on_signals(data.iloc[i], signals)

    # ✅ Execute Trade & Update Portfolio
    trade_action = execute_trade_logic(data.iloc[i], trade_signal, position, cash)

    price = data.iloc[i]["Close"]
    
    if trade_action == "Executing Buy Order":
        buy_amount = cash * 0.5  # Allocate 50% of cash to buying
        position = buy_amount / price
        cash -= buy_amount
        trades.append(f"Buy at {data.iloc[i]['Date']} - {price:.2f}")

    elif trade_action == "Executing Sell Order":
        cash += position * price
        position = 0
        trades.append(f"Sell at {data.iloc[i]['Date']} - {price:.2f}")

    # ✅ Print Portfolio Status
    current_value = cash + (position * price) if position > 0 else cash
    print(f"Date: {data.iloc[i]['Date']}, Portfolio Value: {current_value:.2f}")

# ✅ Final Portfolio Value Calculation
final_value = cash + (position * data.iloc[-1]["Close"]) if position > 0 else cash
print(f"\nFinal Portfolio Value: {final_value:.2f}")
print("Trades Executed:", trades)

print("\nAll Signals:")
for signal in signals:
    print(signal)
