def execute_trade_logic(data, strategy):
    """Execute trading logic based on the strategy signals."""
    signal = strategy(data)
    if signal == "Buy":
        return "Executing Buy Order"
    elif signal == "Sell":
        return "Executing Sell Order"
    else:
        return "Holding"
