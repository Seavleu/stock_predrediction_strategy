import talib as ta

def BBANDS_indicator(data):
    """Bollinger Bands (BBANDS) strategy."""
    upper, middle, lower = ta.BBANDS(data['closing_price'], timeperiod=20)
    
    if data['closing_price'].iloc[-1] > upper.iloc[-1]:
        return "Sell"
    elif data['closing_price'].iloc[-1] < lower.iloc[-1]:
        return "Buy"
    else:
        return "Hold"
