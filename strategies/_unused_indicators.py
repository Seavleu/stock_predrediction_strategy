# ==============================================================================================================
#                                              Momentum Indicators
# ==============================================================================================================
"""
==============================================================================================================
# MACD Indicator: 
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
# RSI Indicator: 
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
    rsi = ta.RSI(data['closing_price'], timeperiod=14)
    if rsi.iloc[-1] < 35:  # 🔹 Reduce Buy threshold from 30 → 35 (more signals)
        return "Buy"
    elif rsi.iloc[-1] > 65:  # 🔹 Reduce Sell threshold from 70 → 65
        return "Sell"
    else:
        return "Hold"



"""
==============================================================================================================
# Stochastic Oscillator Indicator:
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
    slowk, slowd = ta.STOCH(data['highest_price'], data['lowest_price'], data['closing_price'], fastk_period=14, slowk_period=3, slowd_period=3)
    if slowk.iloc[-1] > 80:
        return "Sell"
    elif slowk.iloc[-1] < 20:
        return "Buy"
    else:
        return "Hold"

"""
==============================================================================================================
# Williams %R Indicator:
It ranges from -100 to 0, where values closer to 0 indicate an overbought condition, and values closer to -100 
indicate an oversold condition.

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
# Money Flow Index (MFI) Indicator: 
# - Buy when MFI is below 20 (indicating strong buying pressure)
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
    
# ==============================================================================================================
#                                              Volume_indicators
# ==============================================================================================================
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


# ==============================================================================================================
#                                              Volatility indicators
# ==============================================================================================================
# Bollinger Bands, ATR
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

# ==============================================================================================================
#                                          Statistical Indicators        
# ==============================================================================================================
def BETA_indicator(data):
    """Beta strategy."""
    beta = ta.BETA(data['highest_price'], data['lowest_price'], timeperiod=5)
    if beta.iloc[-1] > 1:
        return "Buy"
    elif beta.iloc[-1] < 1:
        return "Sell"
    else:
        return "Hold"
