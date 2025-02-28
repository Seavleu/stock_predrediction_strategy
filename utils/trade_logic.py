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

# ✅ Predefined Risk Management Constants
STOP_LOSS = 0.97  # Sell if the price drops 3%
TAKE_PROFIT = 1.05  # Sell if the price rises 5%
POSITION_SIZE = 0.5  # Allocate 50% of the portfolio per trade


def execute_trade_logic(data: Dict[str, Any], strategy: str, position: float, cash: float) -> Literal['Executing Buy Order', 'Executing Sell Order', 'Holding']:
    """
    Execute trading logic based on strategy signals.

    Parameters:
    - data: Dictionary containing stock data for the current row.
    - strategy: Trade signal ('Buy', 'Sell', 'Hold').
    - position: The current position size.
    - cash: Available cash balance.

    Returns:
    - A message indicating the action taken.
    """
    price = data['Close']

    if strategy == "Buy" and position == 0:
        # ✅ Buy using POSITION_SIZE allocation
        buy_amount = cash * POSITION_SIZE
        new_position = buy_amount / price
        print(f"Executing Buy Order at {price:.2f}, New Position: {new_position:.4f}")
        return 'Executing Buy Order'

    elif strategy == "Sell" and position > 0:
        # ✅ Sell the existing position
        print(f"Executing Sell Order at {price:.2f}, Closing Position: {position:.4f}")
        return 'Executing Sell Order'

    else:
        print("Holding Position")
        return 'Holding'


def execute_trade_logic_based_on_signals(data: Dict[str, Any], signals: List[Dict[str, str]]) -> str:
    """
    Aggregates trade signals and determines final Buy/Sell/Hold action.

    Parameters:
    - data: Dictionary containing stock data for the current row.
    - signals: List of dictionaries containing signals from multiple indicators.

    Returns:
    - The action to take ('Buy', 'Sell', 'Hold').
    """
    # ✅ Weighted System for Stronger Signals
    weight_map = {
        "BBANDS": 1.2, "EMA": 1.1, "MACD": 1.5, "RSI": 1.3, 
        "ATR": 1.1, "ADX": 1.4, "OBV": 1.2, "SAR": 1.3, 
        "CCI": 1.1, "Fibonacci": 1.0
    }

    buy_score = 0
    sell_score = 0

    for signal in signals:
        for indicator, value in signal.items():
            if value == "Buy":
                buy_score += weight_map.get(indicator, 1)
            elif value == "Sell":
                sell_score += weight_map.get(indicator, 1)

    # ✅ Enhanced Trade Decision Logic (Higher threshold for more reliable trades)
    if buy_score >= 4.0:  # More confident Buy
        print(f"Date: {data['Date']}, Action: Buy, Score: {buy_score:.2f}")
        return "Buy"
    elif sell_score >= 4.0:  # More confident Sell
        print(f"Date: {data['Date']}, Action: Sell, Score: {sell_score:.2f}")
        return "Sell"
    else:
        print(f"Date: {data['Date']}, Action: Hold, Buy Score: {buy_score:.2f}, Sell Score: {sell_score:.2f}")
        return "Hold"


def generate_trade_signal(price_pred: float, macd_signal: str, adx: float, rsi: float, atr: float, sentiment_score: float, confidence: float = 0.8) -> str:
    """
    Generates a trade signal using multiple indicators.

    Parameters:
    - price_pred: Predicted price movement ratio (e.g., 1.02 for +2%).
    - macd_signal: MACD output ('Buy', 'Sell', 'Hold').
    - adx: ADX value (trend strength).
    - rsi: RSI value (momentum indicator).
    - atr: ATR value (volatility measure).
    - sentiment_score: Sentiment analysis output (-1 to +1).
    - confidence: Model confidence in the prediction (default=0.8).

    Returns:
    - 'BUY', 'SELL', or 'HOLD' decision.
    """
    if price_pred > 1.02 and macd_signal == "Buy" and adx > 25 and rsi < 35 and atr > 2:
        return "BUY" if confidence > 0.8 else "HOLD"
    elif price_pred < 0.98 and macd_signal == "Sell" and adx < 20 and rsi > 65 and sentiment_score < -0.5:
        return "SELL"
    else:
        return "HOLD"
