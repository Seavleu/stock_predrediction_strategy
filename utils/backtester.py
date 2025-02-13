import pandas as pd
from strategies.overlap_studies import BBANDS_indicator, EMA_indicator
from strategies.momentum_indicators import MACD_indicator, RSI_indicator, MFI_indicator, WILLR_indicator, CMF_indicator, AROON_indicator, STOCH_indicator, MACDEXT_indicator, AD_indicator
from utils.data_loader import load_korean_stock_data
from utils.trade_logic import execute_trade_logic

def backtest_strategy(data, strategies):
    """Backtest the strategies on the given data."""
    
    portfolio_value = 10000  # Starting amount in the portfolio
    position = 0  # No position initially
    cash = portfolio_value
    trades = []
    signals = []

    for i in range(len(data)):
        # Get signals for each strategy
        signal_bbands = strategies['BBANDS'](data.iloc[i:i+1])
        signal_ema = strategies['EMA'](data.iloc[i:i+1])
        signal_macd = strategies['MACD'](data.iloc[i:i+1])
        signal_rsi = strategies['RSI'](data.iloc[i:i+1])
        signal_mfi = strategies['MFI'](data.iloc[i:i+1])
        signal_willr = strategies['WILLR'](data.iloc[i:i+1])
        signal_cmf = strategies['CMF'](data.iloc[i:i+1])
        signal_aroon = strategies['AROON'](data.iloc[i:i+1])
        signal_stoch = strategies['STOCH'](data.iloc[i:i+1])
        signal_macd_ext = strategies['MACDEXT'](data.iloc[i:i+1])
        signal_ad = strategies['AD'](data.iloc[i:i+1])

        # Append the signals for each date for analysis later
        signals.append({
            'Date': data.iloc[i]['timestamp'],
            'BBANDS': signal_bbands,
            'EMA': signal_ema,
            'MACD': signal_macd,
            'RSI': signal_rsi,
            'MFI': signal_mfi,
            'WILLR': signal_willr,
            'CMF': signal_cmf,
            'AROON': signal_aroon,
            'STOCH': signal_stoch,
            'MACDEXT': signal_macd_ext,
            'AD': signal_ad
        })

        # Simplified execution logic:
        # - Buy if any strategy says "Buy"
        # - Sell if any strategy says "Sell"
        if signal_bbands == "Buy" or signal_ema == "Buy" or signal_macd == "Buy" or signal_rsi == "Buy" or signal_mfi == "Buy" or signal_willr == "Buy" or signal_cmf == "Buy" or signal_aroon == "Buy" or signal_stoch == "Buy" or signal_macd_ext == "Buy" or signal_ad == "Buy":
            if position == 0:  # Buy if no position
                position = cash / data.iloc[i]["closing_price"]
                cash = 0
                trades.append(f"Buy at {data.iloc[i]['timestamp']} - {data.iloc[i]['closing_price']}")

        elif signal_bbands == "Sell" or signal_ema == "Sell" or signal_macd == "Sell" or signal_rsi == "Sell" or signal_mfi == "Sell" or signal_willr == "Sell" or signal_cmf == "Sell" or signal_aroon == "Sell" or signal_stoch == "Sell" or signal_macd_ext == "Sell" or signal_ad == "Sell":
            if position > 0:  # Sell if holding position
                cash = position * data.iloc[i]["closing_price"]
                position = 0
                trades.append(f"Sell at {data.iloc[i]['timestamp']} - {data.iloc[i]['closing_price']}")

        # Print the portfolio status after each transaction
        print(f"Date: {data.iloc[i]['timestamp']}, Portfolio Value: {cash + (position * data.iloc[i]['closing_price'])}")

    # Final portfolio value (cash + any position left)
    final_value = cash + (position * data.iloc[-1]["closing_price"]) if position > 0 else cash
    print(f"\nFinal Portfolio Value: {final_value}")
    print("Trades:", trades)

    return signals


# Load data and run backtest
data = load_korean_stock_data('data/korean_stock_data.csv')  # Load your stock data here
strategies = {
    'BBANDS': BBANDS_indicator,
    'EMA': EMA_indicator,
    'MACD': MACD_indicator,
    'RSI': RSI_indicator,
    'MFI': MFI_indicator,
    'WILLR': WILLR_indicator,
    'CMF': CMF_indicator,
    'AROON': AROON_indicator,
    'STOCH': STOCH_indicator,
    'MACDEXT': MACDEXT_indicator,
    'AD': AD_indicator
}

signals = backtest_strategy(data, strategies)
