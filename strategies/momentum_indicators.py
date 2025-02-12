import talib as ta
import pandas as pd

"""
This indicator solely relies on 'timestamp', 'opening_price', 'highest_price', 
'lowest_price', 'closing_price', 'trading_volume', and 'company'.

For each row of data, we will apply these indicators to run sequentially and make decisions 
based on their outputs. And based on the calculation of these signal we could determine whether 
it is BUY/SELL/HOLD.

For example, if multiple indicator suggest 'BUY', we may decided to go along on the stock. Likewise, 
if it suggests 'SELL' we can decided to short the stock.

To test these indicators, you can run 'python run_simulation.py'

Output example: 
{'Date': Timestamp('2015-10-21 00:00:00'), 'BBANDS': 'Hold', 'EMA': 'Hold', 'MACD': 'Hold', 'RSI': 'Hold'}


Read more here: 
https://vast-part-d09.notion.site/Logs-1987e6ac7d0780dd91c2c975af5f2c2c?pvs=73 
"""


"""
==============================================================================================================
MACD Indicator:
- Buy when MACD crosses above its signal line
- Sell wehn MACD crpsses below its signal line

Example:
    - The stock's MACD line crosses above its signal line, indicating bullish momentum
      -> Decision: **BUY**
    - The MACD line drops below the signal line, indicating bearish momentum
      -> Decision: **SELL**
==============================================================================================================
"""
def MACD_indicator(data):
    """Moving Average Convergence/Divergence (MACD) strategy."""
    macd, macdsignal, _ = ta.MACD(data['closing_price'], fastperiod=12, slowperiod=26, signalperiod=9)
    
    if macd.iloc[-1] > macdsignal.iloc[-1]:
        return "Buy"
    elif macd.iloc[-1] < macdsignal.iloc[-1]:
        return "Sell"
    else:
        return "Hold"

"""
========================================================================================================================================================================= 
RSI Indicator:
- Buy when RSI is below 30 (oversold condition)
- Sell when RSI is above 70 (overbought condition)

Example:
    - The RSI drop < 30 -->  oversold and cold be due for a to its price drop significantly, negative news, economic downturns, or panic selling
        -> Decision : **BUY**
    - The RSI rises > 70 --> overbought and cold be due to intense buying pressure or company's new is particularyl good.
        -> Decision : **SELL**
=========================================================================================================================================================================
"""
def RSI_indicator(data):
    """Relative Strength Index (RSI) strategy."""
    rsi = ta.RSI(data['closing_price'], timeperiod=14)
     
    if rsi.iloc[-1] < 30:  
        return "Buy"
    elif rsi.iloc[-1] > 70:
        return "Sell"
    else:
        return "Hold"

"""
==============================================================================================================
Money Flow Index (MFI) Indicator:
- Buy when MFI is below 20 (indicating strong buying pressure)
- Sell when MFI is below 80 (indicating strong selling pressure)

Example:
    - The MFI drop < 20 --> undervalued and cold be due for a to its price drop significantly, negative news, economic downturns, or panic selling
        -> Decision : **BUY**
    - The RSI rises > 80 --> strong selling presure or an overbought condition
        -> Decision : **SELL**
==============================================================================================================
"""
def MFI_indicator(data):
    mfi = ta.MFI(data['highest_price'], data['lowest_price'], data['closing_price'], data['trading_volume'], timeperiod=14)
    
    if mfi.iloc[-1] < 20:
        return "Buy"
    elif mfi.iloc[-1] > 80:
        return "Sell"
    else:
        return "Hold"

"""
==============================================================================================================
It ranges from -100 to 0, where values closer to 0 indicate an overbought condition, and values closer to -100 
indicate an oversold condition.

Williams %R Indicator:
- Sell when values are closer to 0 (overbought condition).
- Buy when values are closer to -100 (oversold condition).

Example:
    - If > -20, --> overbought condition and potential for a reversal (Reversal: price of a stock changes direction in the opposite of its current trend)
        -> Decision : **SELL**
    - If < -80 --> oversold 
        -> Decision : **BUY**
==============================================================================================================
"""
def WILLR_indicator(data):
    """Williams %R strategy."""
    willr = ta.WILLR(data['highest_price'], data['lowest_price'], data['closing_price'], timeperiod=14)
     
    if willr.iloc[-1] > -20:
        return "Sell"
    elif willr.iloc[-1] < -80:
        return "Buy"
    else:
        return "Hold"

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
Pivot points are used to identify potential support and resistance levels. They can help in making decisions 
about entry points.

- Buy if the closing price is lower than the support level
- Sell if it's above the resistance level

Example:
    - The closing price is below the support level --> a possible reversal from the lower level
      -> Decision: **BUY**
    - The closing price is above the resistance level --> the stock is trending upwards
      -> Decision: **SELL**
==============================================================================================================
"""
def pivot_points(data):
    """Pivot Points strategy."""
    high = data['highest_price'].iloc[-1]
    low = data['lowest_price'].iloc[-1]
    close = data['closing_price'].iloc[-1]
    
    pivot = (high + low + close) / 3
    resistance_1 = (2 * pivot) - low
    support_1 = (2 * pivot) - high
    
    if close > resistance_1:
        return "Sell"
    elif close < support_1:
        return "Buy"
    else:
        return "Hold"
    
"""
==============================================================================================================
Aroon is used to identify the presence and strength of trends. It measures how long it has been since the 
highest high and lowest low over a specified period.

- Buy when Aroon up is above 70 (indicating uptrend).
- Sell when Aroon down is above 70 (indicating downtrend).

Example:
    - Aroon up is above 70 --> strong uptrend. 
      -> Decision: **BUY**
    - Aroon down is above 70 --> strong downtrend. 
      -> Decision: **SELL**
==============================================================================================================
""" 
def AROON_indicator(data):
    """Aroon indicator strategy."""
    aroon_up, aroon_down = ta.AROON(data['highest_price'], data['lowest_price'], timeperiod=14)
    if aroon_up.iloc[-1] > 70:
        return "Buy"
    elif aroon_down.iloc[-1] > 70:
        return "Sell"
    else:
        return "Hold"

"""
==============================================================================================================
The Stochastic Oscillator measures the momentum of a stock and compares the closing price to its range 
over a given period.

- Buy when Stochastic is below 20 (indicating oversold condition).
- Sell when Stochastic is above 80 (indicating overbought condition).

Example:
    - if drops below 20, --> oversold condition. 
      -> Decision: **BUY**
    - If rises above 80, --> overbought condition. 
      -> Decision: **SELL**
==============================================================================================================
"""
def STOCH_indicator(data):
    """Stochastic Oscillator strategy."""
    slowk, slowd = ta.STOCH(data['highest_price'], data['lowest_price'], data['closing_price'], fastk_period=14, slowk_period=3, slowd_period=3)
    if slowk.iloc[-1] > 80:
        return "Sell"
    elif slowk.iloc[-1] < 20:
        return "Buy"
    else:
        return "Hold"


"""
============================================================================================================== 
- If MACD crosses above the signal line, it's a buy signal.
- If MACD crosses below the signal line, it's a sell signal.

Example:
    - Crosses above its signal line --> bullish momentum. 
      -> Decision: **BUY**
    - Crosses below its signal line --> bearish momentum. 
      -> Decision: **SELL**
==============================================================================================================
"""
def MACDEXT_indicator(data):
    """MACD Extended strategy."""
    macd, macdsignal, _ = ta.MACDEXT(data['closing_price'], fastperiod=8, slowperiod=17, signalperiod=9)
    if macd.iloc[-1] > macdsignal.iloc[-1]:
        return "Buy"
    elif macd.iloc[-1] < macdsignal.iloc[-1]:
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

