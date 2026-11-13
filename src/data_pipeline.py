"""
src/data_pipeline.py
Purpose: Market data ingestion & storage using DuckDB & Parquet.
"""

import os
import duckdb
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "research.duckdb")
PARQUET_PATH = os.path.join(DATA_DIR, "market_data.parquet")

TICKER_MAP = {
    "NIFTY_SPOT": "^NSEI",          # Nifty 50 Index
    "BANKNIFTY_SPOT": "^NSEBANK",    # Nifty Bank Index
    "INDIA_VIX": "^INDIAVIX",       # India Volatility Index
    "SP500_FUT": "ES=F",            # US S&P 500 E-mini futures
    "CRUDE_OIL": "CL=F",            # WTI Crude Oil
    "USD_INR": "USDINR=X"           # USD to INR exchange rate
}

def fetch_and_sync_daily_data(lookback_days: int = 730) -> pd.DataFrame:
    """
    Downloads historical EOD data for research tickers, standardizes columns,
    and stores cleanly into DuckDB and Parquet format.
    """
    print(f"🔄 Fetching {lookback_days} days of market data from Yahoo Finance...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    records = []
    for friendly_name, symbol in TICKER_MAP.items():
        try:
            print(f"  -> Downloading {friendly_name} ({symbol})...")
            df = yf.download(symbol, start=start_date, end=end_date, interval="1d", progress=False, auto_adjust=True)
            if df.empty:
                print(f"  ⚠️ Warning: No data returned for {symbol}")
                continue

            # In newer yfinance versions, df.columns might be MultiIndex if single ticker or tuple
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            # Standardize column names to lowercase
            df.columns = [str(c).lower() for c in df.columns]
            
            # Ensure 'date' column is present
            if 'date' not in df.columns and 'datetime' in df.columns:
                df.rename(columns={'datetime': 'date'}, inplace=True)
                
            df['date'] = pd.to_datetime(df['date']).dt.date
            df['symbol'] = friendly_name
            
            # Select required columns
            df_clean = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']].copy()
            df_clean['open'] = df_clean['open'].astype(float)
            df_clean['high'] = df_clean['high'].astype(float)
            df_clean['low'] = df_clean['low'].astype(float)
            df_clean['close'] = df_clean['close'].astype(float)
            df_clean['volume'] = df_clean['volume'].fillna(0).astype(float)
            records.append(df_clean)
        except Exception as e:
            print(f"  ❌ Error downloading {friendly_name}: {e}")

    if not records:
        raise ValueError("Could not download any market data. Check your internet connection.")

    combined = pd.concat(records, ignore_index=True)
    combined = combined.drop_duplicates(subset=['date', 'symbol']).sort_values(['date', 'symbol'])

    # Save to Parquet
    combined.to_parquet(PARQUET_PATH, index=False)
    print(f"📁 Parquet cache saved: {PARQUET_PATH}")

    # Save to DuckDB
    con = duckdb.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS daily_prices (date DATE, symbol VARCHAR, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE);")
    con.execute("DELETE FROM daily_prices;")
    con.execute("INSERT INTO daily_prices SELECT * FROM combined;")
    
    count = con.execute("SELECT count(*) FROM daily_prices;").fetchone()[0]
    distinct_dates = con.execute("SELECT count(distinct date) FROM daily_prices;").fetchone()[0]
    con.close()

    print(f"✅ DuckDB updated! Total rows: {count} across {distinct_dates} trading sessions.")
    return combined

if __name__ == "__main__":
    fetch_and_sync_daily_data()
