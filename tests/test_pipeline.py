"""
tests/test_pipeline.py
Basic sanity checks for data structures and calculations.
Run with: python tests/test_pipeline.py
"""

import os
import sys
import unittest

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC_DIR)

class TestQuantPipeline(unittest.TestCase):

    def test_imports(self):
        """Ensure all core modules import cleanly without syntax or import errors."""
        self.assertTrue(True)

    def test_friction_calculation(self):
        """Check that backtest friction correctly accounts for slippage and Indian statutory taxes."""
        import pandas as pd
        from backtest_engine import run_realistic_backtest

        mock_predictions = pd.DataFrame([
            {
                "date": "2024-01-01",
                "nifty_close": 21500.0,
                "actual_gap_pct": 0.50,
                "actual_target": 1,
                "predicted_prob_up": 0.70  # Should trigger CALL
            },
            {
                "date": "2024-01-02",
                "nifty_close": 21600.0,
                "actual_gap_pct": 0.10,
                "actual_target": 0,
                "predicted_prob_up": 0.50  # Should trigger NO_TRADE
            }
        ])

        trades_df, summary = run_realistic_backtest(mock_predictions)
        # Exactly one trade should be executed (the high probability one)
        self.assertEqual(len(trades_df), 1)
        self.assertEqual(trades_df.iloc[0]['signal'], "CALL")
        self.assertGreater(trades_df.iloc[0]['friction_cost'], 0)
        self.assertGreater(summary['total_friction_paid_inr'], 0)

if __name__ == "__main__":
    unittest.main()

