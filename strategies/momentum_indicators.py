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
    """Compute technical indicators efficiently, optimized for performance."""
    indicators = []

    for name, group in df.groupby("Date"):  # grouping by Date  
        group = group.sort_index()

        # ✅ Short-Term Indicators (Quick trades)
        group["SMA_10"] = ta.SMA(group["Close"], timeperiod=10)
        group["RSI"] = ta.RSI(group["Close"], timeperiod=14)
        group["STOCH_K"], group["STOCH_D"] = ta.STOCH(group["High"], group["Low"], group["Close"])
        group["WILLR"] = ta.WILLR(group["High"], group["Low"], group["Close"])
        group["MFI"] = ta.MFI(group["High"], group["Low"], group["Close"], group["Volume"])
        group["CCI_14"] = ta.CCI(group["High"], group["Low"], group["Close"], timeperiod=14)
        group["Parabolic_SAR"] = ta.SAR(group["High"], group["Low"])

        # ✅ Medium-Term Indicators (Swing trading)
        group["MACD"], group["MACD_signal"], _ = ta.MACD(group["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
        group["Bollinger_Upper"], group["Bollinger_Middle"], group["Bollinger_Lower"] = ta.BBANDS(
            group["Close"], timeperiod=20
        )
        group["ATR"] = ta.ATR(group["High"], group["Low"], group["Close"], timeperiod=14)

        # ✅ Long-Term Indicators (For trend analysis)
        group["EMA_20"] = ta.EMA(group["Close"], timeperiod=20)
        group["EMA_50"] = ta.EMA(group["Close"], timeperiod=50)
        group["ADX"] = ta.ADX(group["High"], group["Low"], group["Close"], timeperiod=14)
        group["OBV"] = ta.OBV(group["Close"], group["Volume"])

        indicators.append(group)

    df = pd.concat(indicators)
    df.dropna(inplace=True)
    
    return df


# ✅ MACD Indicator Strategy
def MACD_indicator(data):
    macd, macdsignal, _ = ta.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    return "Buy" if macd.iloc[-1] > macdsignal.iloc[-1] else "Sell" if macd.iloc[-1] < macdsignal.iloc[-1] else "Hold"


# ✅ RSI Indicator Strategy
def RSI_indicator(data):
    rsi = ta.RSI(data['Close'], timeperiod=14)
    return "Buy" if rsi.iloc[-1] < 35 else "Sell" if rsi.iloc[-1] > 65 else "Hold"


# ✅ Stochastic Oscillator Strategy
def STOCH_indicator(data):
    slowk, slowd = ta.STOCH(data['High'], data['Low'], data['Close'])
    return "Buy" if slowk.iloc[-1] < 20 else "Sell" if slowk.iloc[-1] > 80 else "Hold"


# ✅ Williams %R Strategy
def WILLR_indicator(data):
    willr = ta.WILLR(data['High'], data['Low'], data['Close'], timeperiod=14)
    return "Buy" if willr.iloc[-1] < -80 else "Sell" if willr.iloc[-1] > -20 else "Hold"


# ✅ Money Flow Index (MFI) Strategy
def MFI_indicator(data):
    mfi = ta.MFI(data['High'], data['Low'], data['Close'], data['Volume'], timeperiod=14)
    return "Buy" if mfi.iloc[-1] < 20 else "Sell" if mfi.iloc[-1] > 80 else "Hold"


# ✅ Bollinger Bands (BBANDS) Strategy
def BBANDS_indicator(data):
    upper, middle, lower = ta.BBANDS(data["Close"], timeperiod=20)
    return "Buy" if data["Close"].iloc[-1] < lower.iloc[-1] else "Sell" if data["Close"].iloc[-1] > upper.iloc[-1] else "Hold"


# ✅ Exponential Moving Average (EMA) Strategy
def EMA_indicator(data):
    ema_20 = ta.EMA(data["Close"], timeperiod=20)
    ema_50 = ta.EMA(data["Close"], timeperiod=50)
    return "Buy" if ema_20.iloc[-1] > ema_50.iloc[-1] else "Sell" if ema_20.iloc[-1] < ema_50.iloc[-1] else "Hold"


def ADX_indicator(data):
    """Average Directional Index (ADX) Strategy."""
    adx = ta.ADX(data["High"], data["Low"], data["Close"], timeperiod=14)
    return "Strong Trend" if adx.iloc[-1] > 25 else "Weak Trend"

def ATR_indicator(data):
    """Average True Range (ATR) - Measures volatility"""
    atr = ta.ATR(data["High"], data["Low"], data["Close"], timeperiod=14)
    return atr.iloc[-1]  # ATR is used as a numeric value, not Buy/Sell/Hold

def CCI_indicator(data):
    """Commodity Channel Index (CCI) - Identifies cyclical trends"""
    cci = ta.CCI(data["High"], data["Low"], data["Close"], timeperiod=14)
    return "Buy" if cci.iloc[-1] < -100 else "Sell" if cci.iloc[-1] > 100 else "Hold"

def Parabolic_SAR_indicator(data):
    """Parabolic SAR - Determines trend direction and reversal points"""
    sar = ta.SAR(data["High"], data["Low"], acceleration=0.02, maximum=0.2)
    return "Buy" if data["Close"].iloc[-1] > sar.iloc[-1] else "Sell" if data["Close"].iloc[-1] < sar.iloc[-1] else "Hold"

def OBV_indicator(data):
    """On-Balance Volume (OBV) strategy."""
    obv = ta.OBV(data['Close'], data['Volume'])

    # ✅ Handle edge case where OBV has less than 2 values
    if len(obv) < 2:
        return "Hold"  # Not enough data to decide

    return "Buy" if obv.iloc[-1] > obv.iloc[-2] else "Sell" if obv.iloc[-1] < obv.iloc[-2] else "Hold"

# ✅ Generate Final Trade Signal Based on AI Model & Indicators
def generate_trade_signal(price_pred, macd_signal, rsi, adx, sentiment_score, confidence=0.8):
    """Combine AI model output + technical indicators for final trade signal."""
    if price_pred > 1.02 and macd_signal == "Buy" and adx > 25 and rsi < 35:
        return "BUY" if confidence > 0.8 else "HOLD"
    elif price_pred < 0.98 and macd_signal == "Sell" and adx < 20 and rsi > 65 and sentiment_score < -0.5:
        return "SELL"
    else:
        return "HOLD"
