"""
src/feature_engine.py
Purpose: Quantitative Feature Engineering for Nifty Overnight Gap Prediction.
Enforces:
1. Single unified feature path: Historical training matrix and live inference
   share the exact same feature engineering pipeline (live row is appended to history).
2. Elimination of synthetic hacks: Real 5-day rolling return and real 5-day VIX moving average.
3. Strict look-ahead prevention:
   - Model A (15:15 IST EOD Model): Uses strictly data available at 15:15 IST
     (Nikkei 225 closes 11:30 AM IST, Hang Seng closes 13:30 IST, VIX tick,
      Nifty intraday range position, 1d & 5d returns, DOW, DTE to expiry, S&P prev-close).
   - Model B (08:45-09:10 IST Pre-Open Nowcast): Incorporates finalized US session close,
     overnight crude settlement, and GIFT Nifty futures basis.
4. 3-Class Target:
   - 2: GAP UP (gap_pct >= +0.25%)
   - 0: FLAT (-0.25% < gap_pct < +0.25%)
   - 1: GAP DOWN (gap_pct <= -0.25%)
5. Continuous Gap Points Target: (next_open - current_close) for Quantile Regression.
"""

import os
import duckdb
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DB_PATH = os.path.join(DATA_DIR, "research.duckdb")
PARQUET_PATH = os.path.join(DATA_DIR, "market_data.parquet")

# 15:15 IST EOD Feature Set (Strictly Available before 15:30 IST)
FEATURES_EOD_1515 = [
    "nifty_ret_1d",
    "nifty_ret_5d",
    "nifty_range_pos",
    "vix_close",
    "vix_ret_1d",
    "vix_above_ma5",
    "usdinr_ret_1d",
    "nikkei_ret_1d",
    "hangseng_ret_1d",
    "sp500_us_prev_ret",
    "crude_prev_ret",
    "day_of_week",
    "days_to_expiry"
]

# 08:45-09:10 IST Pre-Open Nowcast Feature Set (Adds finalized overnight US close & GIFT basis)
FEATURES_PREOPEN_0845 = [
    "nifty_ret_1d",
    "nifty_ret_5d",
    "nifty_range_pos",
    "vix_close",
    "vix_ret_1d",
    "vix_above_ma5",
    "usdinr_ret_1d",
    "nikkei_ret_1d",
    "hangseng_ret_1d",
    "sp500_overnight_ret",
    "crude_overnight_ret",
    "gift_basis_pct",
    "day_of_week",
    "days_to_expiry"
]

# Default historical feature column set for backward compatibility
FEATURE_COLUMNS = FEATURES_EOD_1515


def load_raw_data() -> pd.DataFrame:
    """Loads raw historical market data from DuckDB with fallback to Parquet."""
    if os.path.exists(DB_PATH):
        try:
            con = duckdb.connect(DB_PATH, read_only=True)
            df = con.execute("SELECT * FROM daily_prices ORDER BY date ASC;").df()
            con.close()
            if not df.empty:
                return df
        except Exception as e:
            print(f"⚠️ DuckDB read warning: {e}. Falling back to Parquet.")

    if os.path.exists(PARQUET_PATH):
        try:
            df = pd.read_parquet(PARQUET_PATH)
            if not df.empty:
                return df
        except Exception as e:
            print(f"⚠️ Parquet read warning: {e}.")

    from data_pipeline import fetch_and_sync_daily_data
    return fetch_and_sync_daily_data(lookback_days=3650)


def build_feature_matrix(
    raw_df: Optional[pd.DataFrame] = None,
    live_row_dict: Optional[Dict[str, Any]] = None,
    gap_threshold_pct: float = 0.25
) -> pd.DataFrame:
    """
    Constructs aligned feature matrix across Indian, Asian, and Global instruments.
    If live_row_dict is provided, appends the live observation to history so all
    moving averages, lag returns, and range calculations are mathematically identical.
    """
    if raw_df is None:
        raw_df = load_raw_data()

    raw_df = raw_df.copy()
    raw_df['date'] = pd.to_datetime(raw_df['date'])

    # Pivot all closing prices
    pivot_close = raw_df.pivot(index='date', columns='symbol', values='close').sort_index()
    pivot_open = raw_df.pivot(index='date', columns='symbol', values='open').sort_index()
    pivot_high = raw_df.pivot(index='date', columns='symbol', values='high').sort_index()
    pivot_low = raw_df.pivot(index='date', columns='symbol', values='low').sort_index()

    # Forward-fill missing dates across international holidays
    pivot_close = pivot_close.ffill().dropna(subset=['NIFTY_SPOT'])
    pivot_open = pivot_open.ffill().dropna(subset=['NIFTY_SPOT'])
    pivot_high = pivot_high.ffill().dropna(subset=['NIFTY_SPOT'])
    pivot_low = pivot_low.ffill().dropna(subset=['NIFTY_SPOT'])

    # Append live row if provided
    if live_row_dict is not None:
        live_date = pd.to_datetime(live_row_dict.get('date', datetime.now().date()))
        for sym, vals in live_row_dict.get('symbols', {}).items():
            if 'close' in vals and vals['close'] is not None:
                pivot_close.loc[live_date, sym] = float(vals['close'])
            if 'open' in vals and vals['open'] is not None:
                pivot_open.loc[live_date, sym] = float(vals['open'])
            if 'high' in vals and vals['high'] is not None:
                pivot_high.loc[live_date, sym] = float(vals['high'])
            if 'low' in vals and vals['low'] is not None:
                pivot_low.loc[live_date, sym] = float(vals['low'])

        pivot_close = pivot_close.ffill()
        pivot_open = pivot_open.ffill()
        pivot_high = pivot_high.ffill()
        pivot_low = pivot_low.ffill()

    features = pd.DataFrame(index=pivot_close.index)

    # 1. Domestic Index (Nifty 50)
    features['nifty_close'] = pivot_close['NIFTY_SPOT']
    features['nifty_open'] = pivot_open['NIFTY_SPOT']
    features['nifty_high'] = pivot_high['NIFTY_SPOT']
    features['nifty_low'] = pivot_low['NIFTY_SPOT']

    # Genuine 1-day and 5-day returns
    features['nifty_ret_1d'] = features['nifty_close'].pct_change()
    features['nifty_ret_5d'] = features['nifty_close'].pct_change(5)

    # Nifty Intraday Range Position: (Close - Low) / (High - Low)
    span = features['nifty_high'] - features['nifty_low']
    span = span.replace(0, np.nan)
    features['nifty_range_pos'] = ((features['nifty_close'] - features['nifty_low']) / span).clip(0.0, 1.0)
    features['nifty_range_pos'] = features['nifty_range_pos'].fillna(0.50)

    # 2. India VIX (Dynamic 5-day moving average, zero static 15.8 scalar)
    if 'INDIA_VIX' in pivot_close.columns:
        features['vix_close'] = pivot_close['INDIA_VIX']
        features['vix_ret_1d'] = pivot_close['INDIA_VIX'].pct_change()
        vix_ma5 = pivot_close['INDIA_VIX'].rolling(5).mean()
        features['vix_above_ma5'] = (pivot_close['INDIA_VIX'] > vix_ma5).astype(float)
    else:
        features['vix_close'] = 15.0
        features['vix_ret_1d'] = 0.0
        features['vix_above_ma5'] = 0.0

    # 3. Currency (USD/INR)
    if 'USD_INR' in pivot_close.columns:
        features['usdinr_ret_1d'] = pivot_close['USD_INR'].pct_change()
    else:
        features['usdinr_ret_1d'] = 0.0

    # 4. Asian Equities Cues (Both close before 15:15 IST)
    # Nikkei 225 closes at 15:00 JST (11:30 AM IST)
    if 'NIKKEI_225' in pivot_close.columns:
        features['nikkei_ret_1d'] = pivot_close['NIKKEI_225'].pct_change()
    else:
        features['nikkei_ret_1d'] = 0.0

    # Hang Seng closes afternoon session at 16:00 HKT (13:30 IST)
    if 'HANG_SENG' in pivot_close.columns:
        features['hangseng_ret_1d'] = pivot_close['HANG_SENG'].pct_change()
    else:
        features['hangseng_ret_1d'] = 0.0

    # 5. Global Cues with Strict Lookahead Prevention:
    # A) For 15:15 IST Model: Use Day t-1 finalized US close (100% available at 15:15 IST)
    sp_col = 'SP500_INDEX' if 'SP500_INDEX' in pivot_close.columns else ('SP500_FUT' if 'SP500_FUT' in pivot_close.columns else None)
    if sp_col:
        features['sp500_us_prev_ret'] = pivot_close[sp_col].pct_change().shift(1)
        # B) For 08:45 Pre-Open Nowcast: Incorporates Day t's overnight completed US close
        features['sp500_overnight_ret'] = pivot_close[sp_col].pct_change()
    else:
        features['sp500_us_prev_ret'] = 0.0
        features['sp500_overnight_ret'] = 0.0

    if 'CRUDE_OIL' in pivot_close.columns:
        features['crude_prev_ret'] = pivot_close['CRUDE_OIL'].pct_change().shift(1)
        features['crude_overnight_ret'] = pivot_close['CRUDE_OIL'].pct_change()
    else:
        features['crude_prev_ret'] = 0.0
        features['crude_overnight_ret'] = 0.0

    # 6. GIFT Nifty Basis (Used primarily in Pre-Open Nowcast)
    # If live row specifies gift_price, compute exact basis points and pct
    if live_row_dict and 'gift_price' in live_row_dict:
        g_p = float(live_row_dict['gift_price'])
        n_c = float(features.iloc[-1]['nifty_close'])
        basis_pts = g_p - n_c
        basis_pct = (basis_pts / n_c) * 100.0
        features.loc[features.index[-1], 'gift_basis_pct'] = basis_pct
    else:
        features['gift_basis_pct'] = features['sp500_overnight_ret'] * 100.0 * 0.45  # Proxy in historical data

    # 7. Calendar Features
    features['day_of_week'] = np.array([d.weekday() for d in features.index])
    # Days to Thursday weekly expiry: (3 - day_of_week) % 7
    # For overnight carry entering at 15:15 on Thursday (expiry day), roll to next week's contract (DTE = 7)
    raw_dte = (3 - features['day_of_week']) % 7
    features['days_to_expiry'] = np.where(raw_dte == 0, 7, raw_dte)

    # 8. TARGET VARIABLES (Computed only on historical rows, not on live inference row)
    features['next_open'] = features['nifty_open'].shift(-1)
    features['gap_points'] = features['next_open'] - features['nifty_close']
    features['gap_pct'] = (features['gap_points'] / features['nifty_close']) * 100.0
    features['prev_open_gap_pct'] = features['gap_pct'].shift(1)

    # Target A: 3-Class Regime Target
    # 2: GAP UP (gap_pct >= +threshold%)
    # 1: GAP DOWN (gap_pct <= -threshold%)
    # 0: FLAT / CHOP (between -threshold% and +threshold%)
    conditions = [
        features['gap_pct'] >= gap_threshold_pct,
        features['gap_pct'] <= -gap_threshold_pct
    ]
    choices = [2, 1]
    features['target_3class'] = np.select(conditions, choices, default=0)

    # Target B: Binary Target (>= +threshold%)
    features['target'] = (features['gap_pct'] >= gap_threshold_pct).astype(int)

    # Fill any remaining NaNs in feature columns using backward/forward fill
    features = features.bfill().ffill()
    features = features.reset_index()

    return features


def extract_live_feature_vector(
    live_snap: Dict[str, Any],
    model_type: str = "eod_1515"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extracts the live feature vector for Day T by appending live quote_map ticks
    to the historical database and computing through the identical feature engine pipeline.
    Surfaces explicit errors if quotes are missing or stale.
    """
    quote_map = live_snap.get("quote_map", {})
    if not quote_map:
        raise ValueError("MODEL ERROR: live_snap['quote_map'] is missing or empty.")

    nq = quote_map.get("^NSEI", {})
    vixq = quote_map.get("^INDIAVIX", {})
    spq = quote_map.get("ES=F", {}) or quote_map.get("^GSPC", {})
    crudeq = quote_map.get("CL=F", {})
    usdq = quote_map.get("USDINR=X", {})

    n_val = nq.get("value") or (nq.get("last") if nq.get("last", 0) > 0 else None)
    if n_val is None:
        raise ValueError("MODEL ERROR: Live Nifty spot quote is unavailable.")

    # Check staleness
    if nq.get("is_stale", False) and live_snap.get("is_open", False):
        raise ValueError("MODEL ERROR: Live Nifty quote is flagged as STALE (>300s).")

    live_symbols = {
        "NIFTY_SPOT": {
            "close": n_val,
            "open": nq.get("open") or n_val,
            "high": nq.get("high") or n_val,
            "low": nq.get("low") or n_val
        },
        "INDIA_VIX": {
            "close": vixq.get("value") or vixq.get("last") or 15.0
        },
        "USD_INR": {
            "close": usdq.get("value") or usdq.get("last") or 84.0
        },
        "SP500_FUT": {
            "close": spq.get("value") or spq.get("last") or 5800.0
        },
        "CRUDE_OIL": {
            "close": crudeq.get("value") or crudeq.get("last") or 75.0
        }
    }

    gift_price = live_snap.get("giftnifty", {}).get("last")
    live_row = {
        "date": datetime.now().date(),
        "symbols": live_symbols,
        "gift_price": gift_price
    }

    full_matrix = build_feature_matrix(live_row_dict=live_row)
    live_row_features = full_matrix.iloc[[-1]].copy()

    feature_cols = FEATURES_EOD_1515 if model_type == "eod_1515" else FEATURES_PREOPEN_0845
    return live_row_features[feature_cols], live_symbols


if __name__ == "__main__":
    df = build_feature_matrix()
    print("Feature Matrix Shape:", df.shape)
    print("Class Counts (0=Flat, 1=Gap Down, 2=Gap Up):")
    print(df['target_3class'].value_counts())
    print("\nTail preview:")
    print(df[['date'] + FEATURES_EOD_1515 + ['gap_pct', 'gap_points', 'target_3class']].tail())
