# Statistical Indicators (Mean, Z-Score)

import talib as ta

def BETA_indicator(data):
    """Beta strategy."""
    beta = ta.BETA(data['highest_price'], data['lowest_price'], timeperiod=5)
    
    if beta.iloc[-1] > 1:
        return "Buy"
    elif beta.iloc[-1] < 1:
        return "Sell"
    else:
        return "Hold"
