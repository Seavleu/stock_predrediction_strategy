"""
Suggestions:
1. Parameter Type Clarification: You should specify the types of signals and data 
in the function signature for better code clarity and to avoid confusion in the future.

2. Handling Edge Cases: You might want to handle situations where there are no strong 
signals (e.g., all signals are "Hold"). For example, the system could take a wait-and-see 
approach if fewer than three indicators suggest a trade.

3. Additional Strategy Checks: Consider adding more specific conditions for buying 
and selling, such as checking if there is already an open position, risk constraints, or 
portfolio size limitations.
"""
from typing import List, Dict, Literal, Any

# risk management
STOP_LOSS = 0.97  # sell if price drops 3%
TAKE_PROFIT = 1.05  # sell if price rises 5%
POSITION_SIZE = 0.5  # allocate 50% of portfolio per trade

def execute_trade_logic(data: Dict[str, Any], strategy: str, position: float, cash: float, entry_price: float):
    price = data['Close']

    # ✅ stop-Loss & take-Profit Execution (Prioritize forced exits)
    if position > 0:
        if price >= entry_price * TAKE_PROFIT:
            cash += position * price
            print(f"🎯 Take-Profit Triggered: Sold at {price:.2f}, Position Closed, New Cash: {cash:.2f}")
            return 0, cash, 0  # ✅ Reset position

        if price <= entry_price * STOP_LOSS:
            cash += position * price
            print(f"🚨 Stop-Loss Triggered: Sold at {price:.2f}, Position Closed, New Cash: {cash:.2f}")
            return 0, cash, 0  # ✅ Reset position

    # ✅ selling happens when `Sell Score ≥ 3.0`
    if strategy == "Sell" and position > 0:
        cash += position * price  # ✅ Convert position to cash
        print(f"📉 Executing Sell Order at {price:.2f}, Closing Position: {position:.4f}, New Cash: {cash:.2f}")
        return 0, cash, 0  # ✅ Reset position

    elif strategy == "Sell" and position == 0:
        print(f"⚠️ Sell signal detected but no position is open! Skipping trade.")

    # ✅ buy Condition (Only if no position)
    if strategy == "Buy" and position == 0:
        buy_amount = cash * POSITION_SIZE
        new_position = buy_amount / price
        cash -= buy_amount
        print(f"✅ Executing Buy Order at {price:.2f}, New Position: {new_position:.4f}, Cash Left: {cash:.2f}")
        return new_position, cash, price  # ✅ save entry_price

    # ✅ if no buy/sell action, return unchanged values
    print("⚠️ Holding Position")
    return position, cash, entry_price


def execute_trade_logic_based_on_signals(data, signals):
    weight_map = {
        "BBANDS": 1.2, "EMA": 1.1, "MACD": 1.5, "RSI": 1.3, 
        "ATR": 1.1, "ADX": 1.4, "OBV": 1.2, "SAR": 1.3, 
        "CCI": 1.1, "LSTM_Pred": 1.5   
    }

    buy_score, sell_score = 0, 0

    for signal in signals:
        for indicator, value in signal.items():
            if value == "Buy":
                buy_score += weight_map.get(indicator, 1)
            elif value == "Sell":
                sell_score += weight_map.get(indicator, 1)

    # ✅ add LSTM Prediction in Decision Logic
    lstm_pred_price = signals[-1]["LSTM_Pred"]
    actual_close_price = data["Close"]

    if lstm_pred_price > actual_close_price * 1.02:  # Expect 2% increase
        buy_score += 3.0  
    elif lstm_pred_price < actual_close_price * 0.98:  # Expect 2% drop
        sell_score += 3.0  

    if buy_score >= 3.0:  
        print(f"Date: {data['Date']}, Action: Buy, Buy Score: {buy_score:.2f}, Sell Score: {sell_score:.2f}")
        return "Buy"
    elif sell_score >= 3.0:  
        print(f"Date: {data['Date']}, Action: Sell, Buy Score: {buy_score:.2f}, Sell Score: {sell_score:.2f}")
        return "Sell"
    else:
        print(f"Date: {data['Date']}, Action: Hold, Buy Score: {buy_score:.2f}, Sell Score: {sell_score:.2f}")
        return "Hold"

