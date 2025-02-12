from typing import Literal, Any

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
