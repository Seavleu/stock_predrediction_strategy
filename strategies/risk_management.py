# Dynamic stop-loss/take-profit ensures smarter exits.

def calculate_stop_loss(entry_price, volatility, risk_tolerance=0.02):
    """Dynamically calculate stop-loss based on market conditions."""
    return entry_price * (1 - (volatility * risk_tolerance))

def calculate_take_profit(entry_price, volatility, profit_target=0.05):
    """Adjust take-profit dynamically."""
    return entry_price * (1 + (volatility * profit_target))
