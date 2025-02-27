""" Indicators in here consisted of: MACD, RSI, STOCH, WILLR, and MFI.

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

import talib as ta
import pandas as pd


def calculate_technical_indicators(df):
    df['RSI'] = ta.RSI(df['Close'], timeperiod=14)
    df['MACD'], _, _ = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['EMA_20'] = ta.EMA(df['Close'], timeperiod=20)
    df['EMA_50'] = ta.EMA(df['Close'], timeperiod=50)
    df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    return df


def calculate_technical_indicators_2(df):
    """Compute technical indicators efficiently to avoid memory issues."""
    
    # ✅ Process indicators in chunks (grouped by company)
    indicators = []

    for name, group in df.groupby("company"):
        group = group.sort_index()

        # ✅ Apply technical indicators only on relevant columns
        group["SMA_50"] = ta.SMA(group["closing_price"], timeperiod=50)
        group["EMA_20"] = ta.EMA(group["closing_price"], timeperiod=20)
        group["MACD"], group["MACD_signal"], _ = ta.MACD(group["closing_price"], fastperiod=12, slowperiod=26, signalperiod=9)
        group["RSI"] = ta.RSI(group["closing_price"], timeperiod=14)

        indicators.append(group)

    df = pd.concat(indicators)

    # ✅ Drop NaN values after calculations
    df.dropna(inplace=True)

    return df

# Implement decision rules combining technical + AI model outputs
######################################################################################
def generate_trade_signal(price_pred, macd_signal, sentiment_score, confidence=0.8):
    """Generate final trading signal using AI model + technical indicators + sentiment."""
    if price_pred > 1.02 and macd_signal == "Buy" and sentiment_score > 0.5:
        return "BUY" if confidence > 0.8 else "HOLD"
    elif price_pred < 0.98 and macd_signal == "Sell" and sentiment_score < -0.5:
        return "SELL"
    else:
        return "HOLD"

# MACD Indicator:
######################################################################################
"""
==============================================================================================================
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


# RSI Indicator:
######################################################################################
"""
========================================================================================================================================================================= 
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



# Stochastic Oscillator Indicator:
######################################################################################
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
    slowk, slowd = ta.STOCH(data['highest_price'], data['lowest_price'], data['closing_price'], fastk_period=14, slowk_period=3, slowd_period=3)
    if slowk.iloc[-1] > 80:
        return "Sell"
    elif slowk.iloc[-1] < 20:
        return "Buy"
    else:
        return "Hold"


# Williams %R Indicator:
######################################################################################
"""
==============================================================================================================
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
    

# Money Flow Index (MFI) Indicator:
######################################################################################
"""
==============================================================================================================
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