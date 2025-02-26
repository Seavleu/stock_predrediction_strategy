""" Implement your rule-based technical indicators (e.g., moving averages, RSI) in the strategy folder (like momentum_indicators.py or volatility_indicators.py). 
These indicators will generate signals that will later serve as input to the RL model.
"""
import talib as ta   

"""
==============================================================================================================
CMF indicator uses both price and volume to show the amount of money flowing in and out of a security.

- Buy when CMF is greater than 0 (indicating buying pressure).
- Sell when CMF is less than 0 (indicating selling pressure).

Example:
    - If crosses above 0, --> buying pressure from both price and volume
        -> Decision : **BUY**
    - If crosses below 0, --> selling pressure 
        -> Decision : **SELL**
==============================================================================================================
"""
def CMF_indicator(data, timeperiod=20):
    # Calculate Money Flow Multiplier
    mf_multiplier = ((data['closing_price'] - data['lowest_price']) - (data['highest_price'] - data['closing_price'])) / (data['highest_price'] - data['lowest_price'])
    
    # Calculate Money Flow Volume
    mf_volume = mf_multiplier * data['trading_volume']
    
    # Calculate CMF: Moving sum of MF Volume / Moving sum of Volume for the given time period
    cmf = mf_volume.rolling(window=timeperiod).sum() / data['trading_volume'].rolling(window=timeperiod).sum()
    
    # Determine Buy, Sell, Hold based on CMF value
    if cmf.iloc[-1] > 0:
        return "Buy"
    elif cmf.iloc[-1] < 0:
        return "Sell"
    else:
        return "Hold"


"""
==============================================================================================================
The A/D indicator shows the cumulative flow of money into or out of the stock.

- Shows the cumulative flow of money into or out of the stock.
- Buy when current A/D value is greater than the previous one, sell otherwise.

Example:
    - The current A/D value is higher than the previous --> buying pressure
      -> Decision: **BUY**
    - The current A/D value is lower than the previous --> selling pressure
      -> Decision: **SELL**
==============================================================================================================
"""
def AD_indicator(data):
    """Accumulation/Distribution strategy."""
    ad = ta.AD(data['highest_price'], data['lowest_price'], data['closing_price'], data['trading_volume'])
    
    # Ensure there are at least two data points to compare
    if len(ad) > 1:
        if ad.iloc[-1] > ad.iloc[-2]:
            return "Buy"
        else:
            return "Sell"
    else:
        return "Hold"
