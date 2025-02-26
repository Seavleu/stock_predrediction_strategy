# Ensures profitability before deployment
def backtest_strategy(data, model):
    """Simulate trading strategy on historical data."""
    balance = 10000  # Starting balance
    for date, row in data.iterrows():
        price_pred = model.predict(row['features'])
        signal = generate_trade_signal(price_pred, row['MACD'], row['sentiment'])
        
        if signal == "BUY":
            balance *= 1.02  # Assuming 2% gain
        elif signal == "SELL":
            balance *= 0.98  # Assuming 2% loss
    
    return balance

"""The backtester will help simulate the trading strategy using historical data.
We will train the RL agent using the generated signals from the rule-based indicators.
"""
# Williams %R Indicator:
######################################################################################
# import pandas as pd
# from strategies.overlap_studies import BBANDS_indicator, EMA_indicator
# from strategies.momentum_indicators import MACD_indicator, RSI_indicator, STOCH_indicator, WILLR_indicator, MFI_indicator
# from strategies.volume_indicators import CMF_indicator, AD_indicator

# from utils.data_loader import load_korean_stock_data
# from utils.trade_logic import execute_trade_logic

# def backtest_strategy(data, strategies):
#     """Backtest the strategies on the given data with a majority rule for buy/sell signals."""
    
#     portfolio_value = 10000  # Starting amount in the portfolio
#     position = 0  # No position initially
#     cash = portfolio_value
#     trades = []
#     signals = []

#     for i in range(len(data)):
#         # Collect signals for each strategy
#         current_signals = {
#             'BBANDS': strategies['BBANDS'](data.iloc[i:i+1]),
#             'EMA': strategies['EMA'](data.iloc[i:i+1]),
#             'MACD': strategies['MACD'](data.iloc[i:i+1]),
#             'RSI': strategies['RSI'](data.iloc[i:i+1]),
#             'STOCH': strategies['STOCH'](data.iloc[i:i+1]),
#             'MFI': strategies['MFI'](data.iloc[i:i+1]),
#             'WILLR': strategies['WILLR'](data.iloc[i:i+1]),
#             'CMF': strategies['CMF'](data.iloc[i:i+1]),
#             'AD': strategies['AD'](data.iloc[i:i+1])
#         }
        
#         # Aggregate signals (use majority rule)
#         buy_signals = sum(1 for signal in current_signals.values() if signal == "Buy")
#         sell_signals = sum(1 for signal in current_signals.values() if signal == "Sell")

#         # Hold decision based on majority signals
#         if buy_signals > sell_signals:
#             signal = "Buy"
#         elif sell_signals > buy_signals:
#             signal = "Sell"
#         else:
#             signal = "Hold"

#         # Store the signal for analysis later
#         signals.append({'Date': data.iloc[i]['timestamp'], **current_signals, 'Final_Signal': signal})

#         # Execute trading logic based on the final signal
#         if signal == "Buy" and position == 0:  # Buy if no position
#             position = cash / data.iloc[i]["closing_price"]
#             cash = 0
#             trades.append(f"Buy at {data.iloc[i]['timestamp']} - {data.iloc[i]['closing_price']}")

#         elif signal == "Sell" and position > 0:  # Sell if holding position
#             cash = position * data.iloc[i]["closing_price"]
#             position = 0
#             trades.append(f"Sell at {data.iloc[i]['timestamp']} - {data.iloc[i]['closing_price']}")

#         # Print portfolio value after each transaction
#         portfolio_value = cash + (position * data.iloc[i]["closing_price"]) if position > 0 else cash
#         print(f"Date: {data.iloc[i]['timestamp']}, Portfolio Value: {portfolio_value}")

#     # Final portfolio value (cash + any position left)
#     final_value = cash + (position * data.iloc[-1]["closing_price"]) if position > 0 else cash
#     print(f"\nFinal Portfolio Value: {final_value}")
#     print("Trades:", trades)

#     return signals
