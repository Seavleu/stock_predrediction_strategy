# Candlestick pattern strategies  
import talib as ta

def CDLDOJI_indicator(data):
    """Doji pattern strategy."""
    cdl_doji = ta.CDLDOJI(data['opening_price'], data['highest_price'], data['lowest_price'], data['closing_price'])
    
    if cdl_doji.iloc[-1] > 0:
        return "Buy"
    elif cdl_doji.iloc[-1] < 0:
        return "Sell"
    else:
        return "Hold"
