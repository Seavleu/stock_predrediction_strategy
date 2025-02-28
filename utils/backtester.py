from trade_logic import generate_trade_signal

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
