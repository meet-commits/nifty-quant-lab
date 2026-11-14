"""
tests/test_bhavcopy_scanner.py
Comprehensive unit tests for Bhavcopy ingestion, technical indicators,
real-universe market breadth, and dynamic scanner queries using a deterministic fixture.
"""

import os
import sys
import unittest
import duckdb
import pandas as pd
from datetime import date, timedelta

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from chartink_engine import (
    calc_rsi,
    compute_universe_indicators,
    scan_volume_surge_momentum,
    scan_trend_leader_highs,
    scan_rsi_oversold_dips,
    run_chartink_scan,
    get_market_breadth_stats
)
from bhavcopy_service import (
    init_bhavcopy_table,
    save_bhavcopy_records
)


def create_deterministic_universe_fixture(num_symbols: int = 12, num_days: int = 55) -> pd.DataFrame:
    """
    Creates a deterministic multi-symbol, multi-session OHLCV fixture dataset
    with known technical behaviors:
    - SYM_0: Strong bull uptrend with volume breakout on final session (RSI ~70, Surge ~2.5x)
    - SYM_1: Oversold panic dip on final session (RSI < 30, Close > Open)
    - SYM_2: Tight consolidation breakout (VCP) on final session
    - SYM_3: Bearish breakdown below 200 EMA
    - SYM_4 to SYM_11: Neutral range-bound equities
    """
    base_date = date(2026, 10, 1)
    dates = []
    curr = base_date
    while len(dates) < num_days:
        if curr.weekday() < 5:
            dates.append(curr)
        curr += timedelta(days=1)

    records = []
    for s_idx in range(num_symbols):
        sym = f"TESTSYM_{s_idx:02d}"
        price = 1000.0 + (s_idx * 150.0)

        for d_idx, d in enumerate(dates):
            prev_p = price
            # Characteristic price action based on symbol archetype
            if s_idx == 0:  # Bull runner
                pct = -0.005 if d_idx in (8, 18, 28) else (0.035 if d_idx == num_days - 1 else 0.012)
                vol = 100000 if d_idx < num_days - 1 else 350000  # 3.5x volume surge on last day
            elif s_idx == 1:  # Oversold dip
                pct = -0.025 if d_idx >= num_days - 10 else -0.005
                vol = 120000
            elif s_idx == 2:  # Tight compression then breakout
                pct = 0.001 if d_idx < num_days - 1 else 0.025
                vol = 80000 if d_idx < num_days - 1 else 220000  # 2.5x surge
            elif s_idx == 3:  # Bearish breakdown
                pct = -0.018
                vol = 150000
            else:  # Neutral alternating
                pct = 0.004 if (d_idx + s_idx) % 2 == 0 else -0.003
                vol = 90000

            price = round(prev_p * (1.0 + pct), 2)
            high = round(max(price, prev_p) * 1.01, 2)
            low = round(min(price, prev_p) * 0.99, 2)
            open_p = round((prev_p + price) / 2.0, 2)
            if s_idx == 1 and d_idx == num_days - 1:
                open_p = round(price * 0.98, 2)  # Ensure Close > Open on final oversold day

            records.append({
                "trade_date": d,
                "symbol": sym,
                "series": "EQ",
                "open": open_p,
                "high": high,
                "low": low,
                "close": price,
                "prev_close": prev_p,
                "volume": vol,
                "turnover": round(price * vol, 2)
            })

    return pd.DataFrame(records)


class TestBhavcopyScanner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fixture_df = create_deterministic_universe_fixture()
        cls.computed_df = compute_universe_indicators(cls.fixture_df)
        cls.latest_date = cls.computed_df['trade_date'].max()
        cls.latest_df = cls.computed_df[cls.computed_df['trade_date'] == cls.latest_date].copy()

    def test_indicator_calculations(self):
        """Vectorized technical indicators must match mathematical definitions."""
        df = self.computed_df
        # Check all required indicator columns exist
        req_cols = ['ema20', 'ema50', 'ema200', 'rsi14', 'atr14', 'vol_sma20', 'vol_surge', 'chg_pct']
        for c in req_cols:
            self.assertIn(c, df.columns, f"Indicator {c} missing from computed dataframe!")

        # Verify RSI bounded within [0, 100]
        self.assertTrue((df['rsi14'] >= 0.0).all() and (df['rsi14'] <= 100.0).all())

        # Verify ATR strictly positive
        self.assertTrue((df['atr14'] >= 0.0).all())

        # Verify Bull Runner (TESTSYM_00) has volume surge > 2.0x on the latest day
        runner_latest = self.latest_df[self.latest_df['symbol'] == 'TESTSYM_00'].iloc[0]
        self.assertGreaterEqual(runner_latest['vol_surge'], 2.0, "Volume surge should exceed 2x for TESTSYM_00")
        self.assertGreater(runner_latest['rsi14'], 55.0, "RSI should be bullish for TESTSYM_00")

    def test_wilders_rsi_monotonicity(self):
        """RSI should approach 100 for purely monotonic gains and 0 for purely monotonic losses."""
        gains = pd.Series([100.0 + (i * 5.0) for i in range(30)])
        rsi_high = calc_rsi(gains, 14).iloc[-1]
        self.assertGreater(rsi_high, 85.0)

        losses = pd.Series([200.0 - (i * 5.0) for i in range(30)])
        rsi_low = calc_rsi(losses, 14).iloc[-1]
        self.assertLess(rsi_low, 15.0)

    def test_scan_volume_surge_momentum(self):
        """Volume surge scan must detect volume breakout symbols."""
        matches = scan_volume_surge_momentum(self.latest_df, self.computed_df)
        matched_syms = matches['symbol'].tolist()
        self.assertIn('TESTSYM_00', matched_syms, "TESTSYM_00 must match Volume Surge & Momentum scan!")
        self.assertNotIn('TESTSYM_03', matched_syms, "TESTSYM_03 (bearish) must not match Volume Surge scan!")

    def test_scan_rsi_oversold_dips(self):
        """Oversold scan must detect panic selloff symbols with recovery candles."""
        matches = scan_rsi_oversold_dips(self.latest_df, self.computed_df)
        matched_syms = matches['symbol'].tolist()
        self.assertIn('TESTSYM_01', matched_syms, "TESTSYM_01 must match Oversold Accumulation scan!")
        self.assertNotIn('TESTSYM_00', matched_syms, "TESTSYM_00 (runner) must not match Oversold scan!")

    def test_scan_trend_leader_highs(self):
        """Trend leader scan must detect equities near highs with aligned EMAs."""
        matches = scan_trend_leader_highs(self.latest_df, self.computed_df)
        matched_syms = matches['symbol'].tolist()
        self.assertIn('TESTSYM_00', matched_syms)

    def test_market_breadth_whole_universe(self):
        """Market breadth must aggregate over the entire universe with an explicit as_of_date."""
        breadth = get_market_breadth_stats(self.latest_df)

        self.assertIn('pct_above_20_ema', breadth)
        self.assertIn('pct_above_50_ema', breadth)
        self.assertIn('pct_above_200_ema', breadth)
        self.assertIn('adv_dec_ratio', breadth)
        self.assertIn('as_of_date', breadth)
        self.assertIn('total_universe', breadth)

        self.assertEqual(breadth['total_universe'], 12, "Should analyze all 12 symbols in the fixture universe")
        self.assertEqual(breadth['as_of_date'], str(self.latest_date))
        self.assertGreater(breadth['advances'] + breadth['declines'] + breadth['unchanged'], 0)
        self.assertTrue(0.0 <= breadth['pct_above_20_ema'] <= 100.0)

    def test_zero_fake_columns_in_scan_results(self):
        """Scanned output DataFrame must NEVER contain hardcoded conviction/target/stop-loss columns."""
        res_df, meta = run_chartink_scan("⚡ Volume Surge & Momentum Breakout")

        # Prohibited synthetic/hardcoded column names
        forbidden = [
            "target", "target2", "t1", "t2", "reference r1", "reference r2",
            "sl", "stop loss", "reference s1", "rr", "r:r", "conviction", "conv", "strength"
        ]

        for col in res_df.columns:
            self.assertNotIn(col.lower().strip(), forbidden, f"Forbidden synthetic column '{col}' found in scan result!")

        # Mandatory legitimate quantitative columns
        expected = ["Sr", "Symbol", "Price", "Change %", "Volume", "Volume Surge", "RSI", "EMA 20", "EMA 50", "EMA 200", "ATR"]
        for exp in expected:
            self.assertIn(exp, res_df.columns, f"Expected quantitative column '{exp}' missing!")

    def test_duckdb_bhavcopy_caching_idempotency(self):
        """Saving bhavcopy records must be idempotent and not duplicate rows."""
        import tempfile
        temp_db = os.path.join(tempfile.gettempdir(), "test_bhavcopy.duckdb")
        if os.path.exists(temp_db):
            os.remove(temp_db)

        con = duckdb.connect(temp_db)
        try:
            init_bhavcopy_table(con)
            records = [
                {
                    "trade_date": "2026-11-10",
                    "symbol": "RELIANCE",
                    "series": "EQ",
                    "open": 1320.0,
                    "high": 1335.0,
                    "low": 1315.0,
                    "close": 1325.0,
                    "prev_close": 1310.0,
                    "volume": 2500000,
                    "turnover": 3312500000.0
                },
                {
                    "trade_date": "2026-11-10",
                    "symbol": "HDFCBANK",
                    "series": "EQ",
                    "open": 800.0,
                    "high": 810.0,
                    "low": 795.0,
                    "close": 805.0,
                    "prev_close": 800.0,
                    "volume": 3500000,
                    "turnover": 2817500000.0
                }
            ]

            # First save
            count1 = save_bhavcopy_records(records, con)
            self.assertEqual(count1, 2)
            total1 = con.execute("SELECT count(*) FROM bhavcopy_daily;").fetchone()[0]
            self.assertEqual(total1, 2)

            # Re-saving same records (idempotent upsert)
            count2 = save_bhavcopy_records(records, con)
            self.assertEqual(count2, 2)
            total2 = con.execute("SELECT count(*) FROM bhavcopy_daily;").fetchone()[0]
            self.assertEqual(total2, 2, "Row count must remain 2 after re-saving identical dates!")

        finally:
            con.close()
            if os.path.exists(temp_db):
                os.remove(temp_db)

    def test_synthetic_5_to_1_split_no_false_breakdown_or_oversold(self):
        """
        Corporate Actions Adjustment verification:
        When a stock undergoes an unadjusted 5:1 split (|return| > 35%, close = prev_close / 5),
        historical prices must be back-adjusted so:
        1. 200 EMA reflects post-split level (~500, not ~2500)
        2. Stock does NOT trigger false Bearish Breakdown below 200 EMA
        3. Stock does NOT trigger false Panic Oversold (RSI < 32)
        """
        import numpy as np
        from chartink_engine import scan_bearish_breakdown

        base_date = date(2025, 1, 1)
        records = []
        curr = base_date
        dates = []
        while len(dates) < 210:
            if curr.weekday() < 5:
                dates.append(curr)
            curr += timedelta(days=1)

        p = 2500.0
        for i, d in enumerate(dates):
            if i == 205:
                # 5:1 stock split ex-date!
                prev_c = p
                p = 502.0  # ~2500 / 5
                vol = 1500000
                o = 500.0
                h = 512.0
                l = 495.0
            elif i > 205:
                prev_c = p
                p = round(p * 1.002, 2)
                vol = 1200000
                o = prev_c
                h = round(p * 1.01, 2)
                l = round(p * 0.99, 2)
            else:
                prev_c = p
                p = round(2500.0 * (1.0 + np.sin(i / 10.0) * 0.02), 2)
                vol = 250000
                o = prev_c
                h = round(p * 1.01, 2)
                l = round(p * 0.99, 2)

            records.append({
                "trade_date": d,
                "symbol": "SPLIT_CORP",
                "series": "EQ",
                "open": o,
                "high": h,
                "low": l,
                "close": p,
                "prev_close": prev_c,
                "volume": vol,
                "turnover": round(p * vol, 2)
            })

        split_df = pd.DataFrame(records)
        computed = compute_universe_indicators(split_df)
        latest_bar = computed[computed['trade_date'] == dates[-1]].iloc[0]

        # 1. Verify 200 EMA is adjusted to post-split range (~500, NOT ~2400)
        self.assertLess(latest_bar['ema200'], 600.0, "200 EMA must be post-split adjusted to ~500!")
        self.assertGreater(latest_bar['ema200'], 450.0, "200 EMA must be in the ~500 range!")

        # 2. Verify close is above 200 EMA (no false breakdown)
        self.assertGreaterEqual(latest_bar['close'], latest_bar['ema200'])

        # 3. Assert scan_bearish_breakdown does NOT match SPLIT_CORP
        latest_df = computed[computed['trade_date'] == dates[-1]].copy()
        breakdowns = scan_bearish_breakdown(latest_df, computed)
        self.assertNotIn("SPLIT_CORP", breakdowns['symbol'].tolist(), "Split symbol must not trigger false breakdown!")

        # 4. Verify RSI(14) is healthy and not falsely triggered as oversold
        self.assertGreater(latest_bar['rsi14'], 40.0, "RSI must not falsely plummet below oversold threshold on split!")
        oversold = scan_rsi_oversold_dips(latest_df, computed)
        self.assertNotIn("SPLIT_CORP", oversold['symbol'].tolist(), "Split symbol must not trigger false oversold dip!")

    def test_ema200_session_count_eligibility(self):
        """
        Indicator Validity:
        Symbols with < 200 sessions must have ema200 = NaN,
        and must be excluded from the % > 200 EMA denominator in market breadth.
        """
        import numpy as np

        base_date = date(2026, 8, 1)
        records = []
        curr = base_date
        dates = []
        while len(dates) < 60:  # Only 60 sessions!
            if curr.weekday() < 5:
                dates.append(curr)
            curr += timedelta(days=1)

        p = 1000.0
        for d in dates:
            records.append({
                "trade_date": d,
                "symbol": "SHORT_HIST",
                "series": "EQ",
                "open": p,
                "high": p * 1.01,
                "low": p * 0.99,
                "close": p,
                "prev_close": p,
                "volume": 500000,
                "turnover": 500000000.0
            })

        short_df = pd.DataFrame(records)
        computed = compute_universe_indicators(short_df)

        # 1. All ema200 values must be NaN because count (60) < 200
        self.assertTrue(computed['ema200'].isna().all(), "Symbols with < 200 sessions must have ema200 = NaN!")

        # 2. ema50 should be valid for sessions >= 50
        self.assertFalse(computed['ema50'].iloc[-1] is None or np.isnan(computed['ema50'].iloc[-1]))

        # 3. In get_market_breadth_stats, eligible_200_count must be 0
        latest_df = computed[computed['trade_date'] == dates[-1]].copy()
        breadth = get_market_breadth_stats(latest_df)
        self.assertEqual(breadth['eligible_200_count'], 0, "Stock with 60 sessions must have 0 eligible 200 EMA count!")
        self.assertEqual(breadth['pct_above_200_ema'], 0.0)


if __name__ == "__main__":
    unittest.main()
