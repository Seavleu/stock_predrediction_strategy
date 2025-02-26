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

def execute_trade_logic(data: Any, strategy: str) -> Literal['Executing Buy Order', 'Executing Sell Order', 'Holding']:
    """
    Execute trading logic based on the strategy signals.
    
    Parameters:
    - data: The stock data for the current row.
    - strategy: The trade signal ('Buy', 'Sell', 'Hold').
    
    Returns:
    - A message indicating the action taken.
    """
    if strategy == "Buy":
        # Execute buy logic (you can add further checks like cash availability, risk limits, etc.)
        print(f"Executing Buy Order at {data['closing_price']}")
        return 'Executing Buy Order'
    elif strategy == "Sell":
        # Execute sell logic (you can add further checks for open positions, etc.)
        print(f"Executing Sell Order at {data['closing_price']}")
        return 'Executing Sell Order'
    else:
        # Hold the position (no action)
        print("Holding")
        return 'Holding'


def execute_trade_logic_based_on_signals(data: Any, signals: List[Dict[str, str]]) -> str:
    """
    Executes the trade logic based on the aggregated signals from multiple indicators.
    
    Parameters:
    - data: The stock data for the current row.
    - signals: A list of dictionaries containing signals from multiple indicators.
    
    Returns:
    - The action to take ('Buy', 'Sell', 'Hold').
    """
    buy_signals = 0
    sell_signals = 0

    # Count the number of "Buy" and "Sell" signals
    for signal in signals:
        if signal['BBANDS'] == "Buy" or signal['EMA'] == "Buy" or signal['MACD'] == "Buy" or signal['RSI'] == "Buy":
            buy_signals += 1
        elif signal['BBANDS'] == "Sell" or signal['EMA'] == "Sell" or signal['MACD'] == "Sell" or signal['RSI'] == "Sell":
            sell_signals += 1

    # Decide on the action based on the number of signals
    if buy_signals >= 3:                                     #Want to take action when only two strong signals are present, change >= 3 to >= 2
        print(f"Date: {data.index}, Action: Buy")
        return "Buy"
    elif sell_signals >= 3:
        print(f"Date: {data.index}, Action: Sell")
        return "Sell"
    else:
        print(f"Date: {data.index}, Action: Hold")
        return "Hold"


def generate_trade_signal(price_pred, macd_signal, sentiment_score):
    """Combine AI model predictions with technical indicators and sentiment analysis."""
    if price_pred > 1.02 and macd_signal == "Buy" and sentiment_score > 0.5:
        return "BUY"
    elif price_pred < 0.98 and macd_signal == "Sell" and sentiment_score < -0.5:
        return "SELL"
    else:
        return "HOLD"
