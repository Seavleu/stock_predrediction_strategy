import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from strategies.momentum_indicators import (
    MACD_indicator, RSI_indicator, STOCH_indicator, WILLR_indicator, MFI_indicator,
    ADX_indicator, ATR_indicator, CCI_indicator, Parabolic_SAR_indicator, OBV_indicator,
    BBANDS_indicator, EMA_indicator
) 
from utils.data_loader import load_stock_data
from utils.trade_logic import execute_trade_logic_based_on_signals, execute_trade_logic
from core.ai_models.lstm_model import LSTMModel  
from sklearn.preprocessing import MinMaxScaler

MODEL_PATH = "models/best_lstm_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_SIZE = 48
NUM_LAYERS = 2
DROPOUT = 0.6 
LOOKBACK_DAYS = 240 
INPUT_SIZE = 5  # Open, High, Low, Close, Volume

model = LSTMModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

DATA_PATH = "data/kospi/kospi_daily_10y.csv"
data = load_stock_data(DATA_PATH)

# ✅ Normalize the features 
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])

def predict_next_day(data, model, scaler):
    """
    Predicts the next day's closing price using LSTM.

    Parameters:
    - data: Normalized input sequence for LSTM.
    - model: Trained LSTM model.
    - scaler: MinMaxScaler used during training.

    Returns:
    - Predicted next day closing price (in KRW).
    """
    last_sequence = data[-LOOKBACK_DAYS:, :]  
    X_test = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = model(X_test)  # output shape (1,1)

    prediction_np = prediction.cpu().numpy().reshape(-1, 1)  # shape is (1,1)

    close_col_index = list(scaler.feature_names_in_).index("Close")  # get column index
    predicted_close = scaler.inverse_transform(np.repeat(prediction_np, 5, axis=1))[:, close_col_index]

    return predicted_close[0] 


portfolio_value = 10000 
position = 0  
cash = portfolio_value
signals = []
trades = []

for i in range(LOOKBACK_DAYS, len(data) - 1): 
    row = data.iloc[i:i+1]
    
    # ✅ Compute Technical Indicators
    signal_bbands = BBANDS_indicator(row)
    signal_ema = EMA_indicator(row)
    signal_macd = MACD_indicator(row)
    signal_rsi = RSI_indicator(row)
    signal_mfi = MFI_indicator(row)
    signal_willr = WILLR_indicator(row)
    signal_stoch = STOCH_indicator(row)
    signal_adx = ADX_indicator(row)
    signal_atr = ATR_indicator(row)
    signal_cci = CCI_indicator(row)
    signal_sar = Parabolic_SAR_indicator(row)
    signal_obv = OBV_indicator(row) 

    predicted_price = predict_next_day(scaled_data[:i], model, scaler)

    # ✅ Compute actual price movement ratio
    actual_tomorrow = data.iloc[i + 1]['Close']
    price_pred_ratio = predicted_price / data.iloc[i]['Close']

    # ✅ Store signals
    signals.append({
        "Date": data.iloc[i]["Date"],
        "BBANDS": signal_bbands,
        "EMA": signal_ema,
        "MACD": signal_macd,
        "RSI": signal_rsi,
        "MFI": signal_mfi,
        "WILLR": signal_willr,
        "STOCH": signal_stoch,
        "ADX": signal_adx,
        "ATR": signal_atr,
        "CCI": signal_cci,
        "SAR": signal_sar,
        "OBV": signal_obv,
        "LSTM_Pred": price_pred_ratio
    })

    # ✅ Determine Trade Signal
    trade_signal = execute_trade_logic_based_on_signals(data.iloc[i], signals)
    trade_action = execute_trade_logic(data.iloc[i], trade_signal, position, cash, data.iloc[i]['Close'])

    # ✅ Update Portfolio Based on Trade
    price = data.iloc[i]["Close"]
    if trade_action == "Executing Sell Order" and position > 0:
        cash += position * price  # convert position to cash
        position = 0  
        trades.append(f"✅ Sell at {data.iloc[i]['Date']} - {price:.2f}")
        print(f"📉 Sold position at {price:.2f}, New Cash: {cash:.2f}")


    elif trade_action == "Executing Sell Order":
        cash += position * price
        position = 0
        trades.append(f"Sell at {data.iloc[i]['Date']} - {price:.2f}")

    current_value = cash + (position * price) if position > 0 else cash
    print(f"Date: {data.iloc[i]['Date']}, Portfolio Value: {current_value:.2f}")

final_value = cash + (position * data.iloc[-1]["Close"]) if position > 0 else cash
print(f"\nFinal Portfolio Value: {final_value:.2f}")
print("Trades:", trades)

print("\nAll Signals:")
for signal in signals:
    print(signal)
