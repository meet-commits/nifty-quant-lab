"""
tests/test_dashboard_resilience.py
Resilience and UI Truth-in-Data tests for dashboard components and AppTest smoke execution.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
DASH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dashboard"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if DASH_DIR not in sys.path:
    sys.path.insert(0, DASH_DIR)

class TestDashboardResilience(unittest.TestCase):

    def test_components_importable(self):
        """All modular components must import cleanly."""
        from components.status_bar import render_status_bar
        from components.data_health import render_data_health_panel
        from components.gap_forecast_card import render_gap_forecast_card
        from components.hero_runway import render_hero_runway
        from components.intraday_section import render_intraday_section
        from components.constituents_grid import render_constituents_grid
        from components.options_card import render_options_card
        from components.backtest_lab import render_backtest_lab
        from components.chartink_suite import render_events_and_news, render_chartink_pro_suite
        from components.footer import render_footer

        self.assertTrue(callable(render_status_bar))
        self.assertTrue(callable(render_data_health_panel))
        self.assertTrue(callable(render_gap_forecast_card))
        self.assertTrue(callable(render_hero_runway))
        self.assertTrue(callable(render_intraday_section))
        self.assertTrue(callable(render_constituents_grid))
        self.assertTrue(callable(render_options_card))
        self.assertTrue(callable(render_backtest_lab))
        self.assertTrue(callable(render_events_and_news))
        self.assertTrue(callable(render_chartink_pro_suite))
        self.assertTrue(callable(render_footer))

    def test_offline_snapshot_handling(self):
        """When market is offline or snapshot empty, story engine must return clean offline state without crashing."""
        from macro_story_engine import build_comprehensive_market_story
        story = build_comprehensive_market_story({})
        self.assertEqual(story["regime_badge"], "FEED OFFLINE / AWAITING DATA")
        self.assertIn("unavailable", story["headline"].lower())

    def test_truth_in_data_no_synthetic_options(self):
        """Synthetic options profile and circular calculations must be eradicated."""
        import macro_story_engine
        self.assertFalse(hasattr(macro_story_engine, "compute_dynamic_options_profile"),
                         "compute_dynamic_options_profile must be eradicated from macro_story_engine")

    def test_index_specific_weights(self):
        """Bank Nifty constituents must be bank stocks only; weights must have as_of_date."""
        from live_market_service import load_index_weights_map
        weights_map = load_index_weights_map()
        self.assertIn("BANK NIFTY", weights_map)
        self.assertIn("NIFTY 50", weights_map)
        self.assertIn("SENSEX", weights_map)

        bank_stocks = weights_map["BANK NIFTY"]
        self.assertGreaterEqual(len(bank_stocks), 10)
        non_banks = ["RELIANCE", "INFY", "TCS", "ITC", "LT"]
        for s in bank_stocks:
            self.assertNotIn(s["sym"], non_banks, f"Non-bank stock {s['sym']} found in Bank Nifty!")
            self.assertTrue(s.get("as_of_date"), f"Constituent {s['sym']} missing as_of_date!")

    def test_neutral_terminology(self):
        """Hero runway file must not contain trading calls like 'STOP LOSS' or 'Dip Buy Zone'."""
        hero_file = os.path.join(DASH_DIR, "components", "hero_runway.py")
        with open(hero_file, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertNotIn("STOP LOSS", content, "'STOP LOSS' should be replaced by 'REFERENCE S1'")
        self.assertNotIn("Dip Buy Zone", content, "'Dip Buy Zone' should be replaced by 'Central Mean Level'")
        self.assertIn("REFERENCE S1", content)
        self.assertIn("PIVOT (P)", content)
        self.assertIn("REFERENCE R1", content)
        self.assertIn("REFERENCE R2", content)

    def test_apptest_smoke(self):
        """AppTest smoke test: dashboard/app.py starts cleanly without unhandled exceptions."""
        from streamlit.testing.v1 import AppTest
        app_path = os.path.join(DASH_DIR, "app.py")
        at = AppTest.from_file(app_path, default_timeout=60)
        at.run()
        self.assertEqual(len(at.exception), 0, f"AppTest raised exceptions: {at.exception}")


if __name__ == "__main__":
    unittest.main()
