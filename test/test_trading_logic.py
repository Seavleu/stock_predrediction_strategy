import unittest
from utils.trade_logic import execute_trade_logic
from strategies.overlap_studies import BBANDS_indicator

class TestTradingLogic(unittest.TestCase):

    def test_trade_logic(self):
        data = pd.DataFrame({
            'timestamp': ['2025-02-11'],
            'closing_price': [336.67]
        })
        result = execute_trade_logic(data, BBANDS_indicator)
        self.assertEqual(result, "Buy")  # Based on the mock data and BBANDS logic
