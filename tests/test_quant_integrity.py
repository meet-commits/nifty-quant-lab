"""
tests/test_quant_integrity.py
Comprehensive quantitative data integrity and truth-in-data test suite.
Validates:
1. Floor pivot mathematical precision from previous session OHLC
2. IST timezone market status and NSE holiday calendar enforcement
3. Exchange symbol to DuckDB table mapping (including Sensex fix)
4. Quote staleness detection (>300s during market hours) and formatting
5. Zero synthetic fallback numbers (honest None / N/A handling)
6. News service resilience (returns empty list on failure, zero synthetic seeds)
"""

import os
import sys
import unittest
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from unittest.mock import patch, MagicMock

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from datapoint import Quote, make_quote
from live_market_service import (
    calc_floor_pivots,
    get_market_session_status,
    load_nse_holidays_map,
    SYMBOL_TO_DB_MAP,
    fetch_nse_gift_nifty,
    IST
)
from news_service import fetch_live_market_news


class TestQuantIntegrity(unittest.TestCase):

    def test_floor_pivots_mathematical_precision(self):
        """Verify classical floor pivots are computed accurately from previous session High, Low, Close."""
        # Using completed session values: H = 25246.35, L = 25054.40, C = 25144.30
        h, l, c = 25246.35, 25054.40, 25144.30
        pivots = calc_floor_pivots(h, l, c)

        # Expected:
        # P = (25246.35 + 25054.40 + 25144.30) / 3 = 25148.35 -> 25148.4
        # S1 = 2 * 25148.35 - 25246.35 = 25050.35 -> 25050.4
        # R1 = 2 * 25148.35 - 25054.40 = 25242.30 -> 25242.3
        # R2 = 25148.35 + (25246.35 - 25054.40) = 25340.30 -> 25340.3
        self.assertEqual(pivots["pivot"], 25148.4)
        self.assertEqual(pivots["s1"], 25050.4)
        self.assertEqual(pivots["sl"], 25050.4)  # sl alias
        self.assertEqual(pivots["r1"], 25242.3)
        self.assertEqual(pivots["t1"], 25242.3)  # t1 alias
        self.assertEqual(pivots["r2"], 25340.3)
        self.assertEqual(pivots["t2"], 25340.3)  # t2 alias

        # Test edge case: None or zero input must return None for all levels
        none_pivots = calc_floor_pivots(None, 25000.0, 25100.0)
        self.assertIsNone(none_pivots["pivot"])
        self.assertIsNone(none_pivots["s1"])
        self.assertIsNone(none_pivots["r1"])

    def test_symbol_to_db_mapping(self):
        """Verify symbols correctly map to DuckDB tables, specifically ^BSESN to SENSEX_SPOT."""
        self.assertEqual(SYMBOL_TO_DB_MAP.get("^NSEI"), "NIFTY_SPOT")
        self.assertEqual(SYMBOL_TO_DB_MAP.get("^NSEBANK"), "BANKNIFTY_SPOT")
        self.assertEqual(SYMBOL_TO_DB_MAP.get("^BSESN"), "SENSEX_SPOT")  # Bug fix check
        self.assertNotEqual(SYMBOL_TO_DB_MAP.get("^BSESN"), "NIFTY_SPOT")
        self.assertEqual(SYMBOL_TO_DB_MAP.get("CL=F"), "CRUDE_OIL")
        self.assertEqual(SYMBOL_TO_DB_MAP.get("USDINR=X"), "USD_INR")
        self.assertEqual(SYMBOL_TO_DB_MAP.get("ES=F"), "SP500_FUT")

    def test_quote_stale_detection(self):
        """Verify Quote strictly flags age > 300 seconds as stale during market hours."""
        now_ist = datetime.now(IST)

        # Fresh quote: 10 seconds ago
        fresh_ts = now_ist - timedelta(seconds=10)
        q_fresh = make_quote(
            value=25144.3,
            ts_ist=fresh_ts,
            source="LIVE_TEST",
            is_market_hours=True
        )
        self.assertFalse(q_fresh.is_stale)
        self.assertTrue(q_fresh.is_valid)
        self.assertEqual(q_fresh.format_value(), "25,144.30")

        # Stale quote: 350 seconds ago (> 300s threshold)
        stale_ts = now_ist - timedelta(seconds=350)
        q_stale = make_quote(
            value=25144.3,
            ts_ist=stale_ts,
            source="LIVE_TEST",
            is_market_hours=True
        )
        self.assertTrue(q_stale.is_stale)
        self.assertFalse(q_fresh.is_stale and False)
        self.assertFalse(q_stale.is_valid)

        # Unavailable quote: value is None
        q_none = make_quote(
            value=None,
            ts_ist=now_ist,
            source="UNAVAILABLE",
            is_market_hours=True
        )
        self.assertIsNone(q_none.value)
        self.assertEqual(q_none.format_value(), "N/A")
        self.assertEqual(q_none.format_pct(), "N/A")
        self.assertEqual(q_none.format_chg(), "N/A")
        d = q_none.to_dict()
        self.assertEqual(d["last"], 0.0)
        self.assertIsNone(d["value"])
        self.assertFalse(d["is_live"])

    def test_nse_holidays_calendar(self):
        """Verify NSE holidays map loads from config/nse_holidays.csv and correctly flags holidays."""
        holidays = load_nse_holidays_map()
        self.assertGreater(len(holidays), 10)
        # Check known Indian exchange holidays
        self.assertIn("2025-01-26", holidays)  # Republic Day
        self.assertIn("2025-08-15", holidays)  # Independence Day
        self.assertIn("2025-10-02", holidays)  # Mahatma Gandhi Jayanti

    def test_market_session_status_holidays_and_weekends(self):
        """Verify market status accurately detects holidays, weekends, and live hours in IST."""
        # Test holiday detection
        with patch("live_market_service.datetime") as mock_dt:
            # 2025-08-15 is Friday (Independence Day)
            mock_dt.now.return_value = datetime(2025, 8, 15, 10, 30, tzinfo=IST)
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            is_open, status_text, color = get_market_session_status()
            self.assertFalse(is_open)
            self.assertIn("EXCHANGE HOLIDAY", status_text)

        # Test weekend detection
        with patch("live_market_service.datetime") as mock_dt:
            # 2026-11-14 is Saturday
            mock_dt.now.return_value = datetime(2026, 11, 14, 11, 0, tzinfo=IST)
            is_open, status_text, color = get_market_session_status()
            self.assertFalse(is_open)
            self.assertIn("WEEKEND CLOSURE", status_text)

        # Test live weekday market hours (e.g. Wednesday 11:00 AM IST)
        with patch("live_market_service.datetime") as mock_dt:
            # 2025-08-20 is Wednesday (normal trading day)
            mock_dt.now.return_value = datetime(2025, 8, 20, 11, 0, tzinfo=IST)
            is_open, status_text, color = get_market_session_status()
            self.assertTrue(is_open)
            self.assertIn("LIVE TRADING OPEN", status_text)

    def test_zero_fake_fallback_for_gift_nifty(self):
        """Verify fetch_nse_gift_nifty never returns fake +28.5 basis or +-35 high/low when offline."""
        with patch("live_market_service.requests.Session") as mock_sess:
            instance = mock_sess.return_value
            instance.get.side_effect = Exception("Connection Refused")

            # Force clear cache
            import live_market_service
            live_market_service._GIFT_CACHE["last_time"] = 0.0
            live_market_service._GIFT_CACHE["data"] = None

            res = fetch_nse_gift_nifty(spot_nifty=25144.3)
            self.assertFalse(res["is_real"])
            self.assertFalse(res["is_live"])
            self.assertIsNone(res["last"])
            self.assertIsNone(res["high"])
            self.assertIsNone(res["low"])
            self.assertEqual(res["chg_str"], "N/A")
            self.assertEqual(res["source"], "UNAVAILABLE")

    def test_news_service_zero_fake_headlines(self):
        """Verify news service returns empty list on RSS failure, never fake synthetic seed articles."""
        if hasattr(fetch_live_market_news, "clear"):
            fetch_live_market_news.clear()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = Exception("DNS Resolution Failure")
            news = fetch_live_market_news(limit=5)
            self.assertEqual(news, [])
            # Assert no fake hardcoded articles
            for n in news:
                self.assertNotIn("Reuters", n.get("source", ""))


if __name__ == "__main__":
    unittest.main()
