import sys
import os
import math
import unittest

# allow tests to import package modules when run from repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from risk_manager import RiskManager


class TestRiskManagerSizing(unittest.TestCase):
    def test_normal_sizing(self):
        rm = RiskManager()
        balance = 10_000.0
        free_margin = 9_000.0
        price = 1.1000
        atr = 0.0015
        res = rm.calculate_position_size(balance, free_margin, price, atr, symbol='EURUSD')
        # basic sanity checks
        self.assertIn('lots', res)
        self.assertGreaterEqual(res['lots'], 0.0)
        # margin required should not exceed free margin
        self.assertLessEqual(res['margin_required'], free_margin + 1e-6)

    def test_reject_below_min(self):
        rm = RiskManager()
        balance = 100.0
        free_margin = 10.0
        price = 1.1000
        atr = 1e-6  # extremely small ATR -> tiny lots
        res = rm.calculate_position_size(balance, free_margin, price, atr, symbol='EURUSD', volume_min=0.01)
        # should reject because computed lots < volume_min
        self.assertEqual(res['lots'], 0.0)
        self.assertTrue(res.get('rejected_due_to_min', False))

    def test_jpy_pair_pip_value(self):
        rm = RiskManager()
        price = 110.00
        pip_value = rm.pip_value_usd('USDJPY', price, lots=1.0)
        # For USDJPY approx: (0.01 / price) * contract_size * lots
        expected = (0.01 / price) * rm.contract_size * 1.0
        self.assertTrue(math.isclose(pip_value, expected, rel_tol=1e-6))


if __name__ == '__main__':
    unittest.main()
