import pandas as pd
from strategies.overlap_studies import BBANDS_indicator

def backtest_strategy(data, strategies):
    """Backtest the strategies on the given data."""
    
    portfolio = 10000  # Starting amount in the portfolio
    position = 0  # No position initially
    cash = portfolio
    trades = []

    for i in range(len(data)):
        # Get signals for each strategy
        signal_bbands = strategies['BBANDS'](data.iloc[i:i+1])
        signal_ema = strategies['EMA'](data.iloc[i:i+1])
        signal_macd = strategies['MACD'](data.iloc[i:i+1])
        
        # Simplified logic for execution:
        # - Buy if any strategy says "Buy"
        # - Sell if any strategy says "Sell"
        
        if signal_bbands == "Buy" or signal_ema == "Buy" or signal_macd == "Buy":
            if position == 0:  # Buy if no position
                position = cash / data.iloc[i]["closing_price"]
                cash = 0
                trades.append(f"Buy at {data.iloc[i]['Date']} - {data.iloc[i]['closing_price']}")
        
        elif signal_bbands == "Sell" or signal_ema == "Sell" or signal_macd == "Sell":
            if position > 0:  # Sell if holding position
                cash = position * data.iloc[i]["closing_price"]
                position = 0
                trades.append(f"Sell at {data.iloc[i]['Date']} - {data.iloc[i]['closing_price']}")

    # Final portfolio value (cash + any position left)
    final_value = cash + (position * data.iloc[-1]["closing_price"]) if position > 0 else cash
    print(f"Final Portfolio Value: {final_value}")
    print("Trades:", trades)


