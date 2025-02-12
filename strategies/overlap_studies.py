import talib as ta

def BBANDS_indicator(data):
    """Bollinger Bands strategy."""
    upper, middle, lower = ta.BBANDS(data['closing_price'], timeperiod=20)
    
    """Buy when price is below the lower band, Sell when price is above the upper band."""
    if data['closing_price'].iloc[-1] > upper.iloc[-1]:
        return "Sell"
    elif data['closing_price'].iloc[-1] < lower.iloc[-1]:
        return "Buy"
    else:
        return "Hold"

def EMA_indicator(data):
    """Exponential Moving Average (EMA) strategy."""
    ema = ta.EMA(data['closing_price'], timeperiod=30)
    
    """Buy when price is above the EMA, Sell when price is below."""
    if data['closing_price'].iloc[-1] > ema.iloc[-1]:
        return "Buy"
    elif data['closing_price'].iloc[-1] < ema.iloc[-1]:
        return "Sell"
    else:
        return "Hold"
