def calculate_vwap(data):
    vwap = (data['closing_price'] * data['num_of_shares']).sum() / data['num_of_shares'].sum()
    return vwap


def liquidity_indicator(data):
    average_shares = data['num_of_shares'].mean()
    if data['num_of_shares'].iloc[-1] > average_shares * 1.5:  # More than 1.5x average
        return "High Liquidity"
    else:
        return "Low Liquidity"


def price_impact_indicator(data):
    # Measure the relative size of the trading activity
    impact = data['num_of_shares'] / data['trading_volume']
    if impact > 0.5:
        return "Strong Price Impact"
    else:
        return "Weak Price Impact"
