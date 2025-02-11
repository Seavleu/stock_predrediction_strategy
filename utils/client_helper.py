from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

def create_trading_client(api_key, api_secret):
    return TradingClient(api_key, api_secret)

def place_order(client, symbol, qty, side=OrderSide.BUY):
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY
    )
    return client.submit_order(order)
