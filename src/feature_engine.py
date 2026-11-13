"""
src/feature_engine.py
Purpose: Quantitative feature engineering for Nifty overnight gap prediction.
Features are calculated strictly without lookahead bias.
"""

import os
import duckdb
import numpy as np
import pandas as pd

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DB_PATH = os.path.join(DATA_DIR, "research.duckdb")
PARQUET_PATH = os.path.join(DATA_DIR, "market_data.parquet")

FEATURE_COLUMNS = [
    "nifty_ret_1d",
    "nifty_ret_5d",
    "nifty_range_pos",
    "vix_ret_1d",
    "vix_above_ma5",
    "sp500_ret_1d",
    "crude_ret_1d",
    "usdinr_ret_1d"
]

def load_raw_data() -> pd.DataFrame:
    """
    Loads raw prices from DuckDB with a fallback to Parquet file,
    or automatically synchronizes fresh data if running on a fresh container.
    """
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

    # Failsafe: Sync initial historical data on fresh cloud containers
    try:
        from data_pipeline import fetch_and_sync_daily_data
        print("🔄 Cold start detected on cloud: auto-syncing historical market archive...")
        return fetch_and_sync_daily_data(lookback_days=500)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load or initialize market data: {e}")

def build_feature_matrix(gap_threshold_pct: float = 0.20) -> pd.DataFrame:
    """
    Constructs an aligned feature matrix across Indian and global instruments.
    Strictly prevents lookahead bias.
    """
    raw_df = load_raw_data()
    raw_df['date'] = pd.to_datetime(raw_df['date'])

    # Separate Nifty OHLC to calculate intraday position & gap target
    nifty_df = raw_df[raw_df['symbol'] == 'NIFTY_SPOT'].copy()
    nifty_df = nifty_df.sort_values('date').reset_index(drop=True)

    # 1. Nifty Intraday Range Position (Stochastic Close Position)
    # Formula: (Close - Low) / (High - Low)
    # Identifies whether institutions closed the market at the top or bottom of the day
    high_low_span = nifty_df['high'] - nifty_df['low']
    high_low_span = high_low_span.replace(0, np.nan)
    nifty_df['nifty_range_pos'] = (nifty_df['close'] - nifty_df['low']) / high_low_span

    # Pivot all closing prices to align instruments by date
    pivot_close = raw_df.pivot(index='date', columns='symbol', values='close')
    pivot_close = pivot_close.ffill().dropna()

    # Calculate returns for each asset
    features = pd.DataFrame(index=pivot_close.index)
    
    # Nifty features
    features['nifty_close'] = pivot_close['NIFTY_SPOT']
    features['nifty_ret_1d'] = pivot_close['NIFTY_SPOT'].pct_change()
    features['nifty_ret_5d'] = pivot_close['NIFTY_SPOT'].pct_change(5)
    
    # Merge range position
    features = features.join(nifty_df.set_index('date')[['nifty_range_pos', 'open', 'high', 'low']], how='inner')
    features = features.rename(columns={'open': 'nifty_open', 'high': 'nifty_high', 'low': 'nifty_low'})

    # India VIX features
    if 'INDIA_VIX' in pivot_close.columns:
        features['vix_close'] = pivot_close['INDIA_VIX']
        features['vix_ret_1d'] = pivot_close['INDIA_VIX'].pct_change()
        vix_ma5 = pivot_close['INDIA_VIX'].rolling(5).mean()
        features['vix_above_ma5'] = (pivot_close['INDIA_VIX'] > vix_ma5).astype(float)
    else:
        features['vix_ret_1d'] = 0.0
        features['vix_above_ma5'] = 0.0

    # Overnight Global Cues
    # Note: ES=F (S&P 500 Futures) and CL=F (Crude) reflect overnight US moves relative to India
    if 'SP500_FUT' in pivot_close.columns:
        features['sp500_ret_1d'] = pivot_close['SP500_FUT'].pct_change()
    else:
        features['sp500_ret_1d'] = 0.0

    if 'CRUDE_OIL' in pivot_close.columns:
        features['crude_ret_1d'] = pivot_close['CRUDE_OIL'].pct_change()
    else:
        features['crude_ret_1d'] = 0.0

    if 'USD_INR' in pivot_close.columns:
        features['usdinr_ret_1d'] = pivot_close['USD_INR'].pct_change()
    else:
        features['usdinr_ret_1d'] = 0.0

    # TARGET VARIABLE:
    # Next Trading Session's Opening Gap = (Next Open - Current Close) / Current Close * 100
    features['next_open'] = features['nifty_open'].shift(-1)
    features['gap_pct'] = (features['next_open'] - features['nifty_close']) / features['nifty_close'] * 100.0

    # Binary Classification Target:
    # 1: Meaningful Gap-Up (>= +threshold%)
    # 0: Gap-Down or Flat (< +threshold%)
    features['target'] = (features['gap_pct'] >= gap_threshold_pct).astype(int)

    # Clean dataset (drop rows with NaN due to rolling calculations)
    clean_df = features.dropna().reset_index()
    return clean_df

if __name__ == "__main__":
    df = build_feature_matrix()
    print("Feature Matrix Preview:")
    print(df[['date'] + FEATURE_COLUMNS + ['gap_pct', 'target']].tail())
