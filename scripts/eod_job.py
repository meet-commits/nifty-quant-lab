"""
scripts/eod_job.py
Automated End-Of-Day (EOD) Market Data Archival Engine.
Runs strictly after 15:45 IST on trading days (or with --force).
Fetches final completed daily bars from official sources / Yahoo Finance,
idempotently writes to DuckDB daily_prices, and updates data/market_data.parquet.
Includes structured logging and zero silent exceptions.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, date, time as dtime
from zoneinfo import ZoneInfo
import pandas as pd
import duckdb
import yfinance as yf

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("EOD_JOB")

IST = ZoneInfo("Asia/Kolkata")
MARKET_CLOSE_TIME = dtime(15, 45)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
DB_PATH = os.path.join(DATA_DIR, "research.duckdb")
PARQUET_PATH = os.path.join(DATA_DIR, "market_data.parquet")
HOLIDAYS_PATH = os.path.join(CONFIG_DIR, "nse_holidays.csv")

TARGET_SYMBOLS = {
    "NIFTY_SPOT": "^NSEI",
    "BANKNIFTY_SPOT": "^NSEBANK",
    "SENSEX_SPOT": "^BSESN",
    "INDIA_VIX": "^INDIAVIX",
    "SP500_FUT": "ES=F",
    "CRUDE_OIL": "CL=F",
    "USD_INR": "USDINR=X"
}


def load_nse_holidays() -> set:
    """Loads NSE holiday dates from config/nse_holidays.csv."""
    holidays = set()
    if os.path.exists(HOLIDAYS_PATH):
        try:
            df = pd.read_csv(HOLIDAYS_PATH)
            for d in df["date"].dropna():
                holidays.add(str(d).strip())
        except Exception as e:
            logger.warning(f"Could not load holidays CSV: {e}")
    return holidays


def is_active_trading_day(target_date: date) -> tuple[bool, str]:
    """Determines if the given date is a valid Indian equity market trading session."""
    weekday = target_date.weekday()
    if weekday in (5, 6):
        return False, f"Weekend closure ({target_date.strftime('%A')})"

    date_str = target_date.strftime("%Y-%m-%d")
    holidays = load_nse_holidays()
    if date_str in holidays:
        return False, f"NSE Holiday calendar match on {date_str}"

    return True, "Active Trading Day"


def fetch_final_daily_candle(friendly_sym: str, yf_sym: str, target_date: date) -> dict:
    """
    Downloads final completed daily bar for the given session.
    Returns normalized OHLCV dict or raises ValueError if not available.
    """
    logger.info(f"Downloading EOD bar for {friendly_sym} ({yf_sym})...")
    ticker = yf.Ticker(yf_sym)
    # Pull 5 days to safely locate target date's finalized bar
    df = ticker.history(period="5d", interval="1d")
    if df.empty:
        raise ValueError(f"No history returned by Yahoo Finance for {yf_sym}")

    df = df.reset_index()
    dt_col = "Date" if "Date" in df.columns else df.columns[0]
    df["session_date"] = pd.to_datetime(df[dt_col]).dt.date

    # Match target date, or if weekend/after-hours take the most recent completed trading date
    match_df = df[df["session_date"] == target_date]
    if match_df.empty:
        latest_date = df["session_date"].max()
        logger.warning(f"No exact bar on {target_date} for {friendly_sym}; using latest completed session: {latest_date}")
        row = df[df["session_date"] == latest_date].iloc[-1]
        session_actual_date = latest_date
    else:
        row = match_df.iloc[-1]
        session_actual_date = target_date

    # Ensure no NaN values
    o = float(row["Open"])
    h = float(row["High"])
    l = float(row["Low"])
    c = float(row["Close"])
    v = float(row.get("Volume", 0.0) or 0.0)

    if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c):
        raise ValueError(f"NaN values found in EOD candle for {friendly_sym} on {session_actual_date}")

    return {
        "date": session_actual_date,
        "symbol": friendly_sym,
        "open": round(o, 2),
        "high": round(h, 2),
        "low": round(l, 2),
        "close": round(c, 2),
        "volume": round(v, 1)
    }


def run_eod_pipeline(force: bool = False, target_date_str: str = None) -> bool:
    """Executes the EOD archival pipeline with safety validations."""
    now_ist = datetime.now(IST)
    logger.info(f"Starting EOD archival pipeline at {now_ist.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    if target_date_str:
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    else:
        target_date = now_ist.date()

    # Time validation (must be >= 15:45 IST on live trading days)
    if not force:
        is_trading, reason = is_active_trading_day(target_date)
        if not is_trading:
            logger.info(f"Skipping EOD job: {reason}. (Use --force to override)")
            return True

        if now_ist.time() < MARKET_CLOSE_TIME:
            logger.error(f"Cannot run EOD job before 15:45 IST (Current: {now_ist.strftime('%H:%M:%S IST')}). Pass --force to override.")
            return False

    os.makedirs(DATA_DIR, exist_ok=True)
    eod_records = []

    for friendly_sym, yf_sym in TARGET_SYMBOLS.items():
        try:
            candle = fetch_final_daily_candle(friendly_sym, yf_sym, target_date)
            eod_records.append(candle)
            logger.info(f"✅ {friendly_sym} finalized: O={candle['open']}, H={candle['high']}, L={candle['low']}, C={candle['close']}")
        except Exception as e:
            logger.error(f"❌ Failed to ingest {friendly_sym}: {e}")

    if not eod_records:
        logger.error("No valid EOD records acquired. Aborting database sync.")
        return False

    # Idempotent write to DuckDB
    logger.info(f"Connecting to DuckDB: {DB_PATH}")
    con = duckdb.connect(DB_PATH)
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS daily_prices (
                date DATE,
                symbol VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE
            );
        """)

        for rec in eod_records:
            # Delete any existing row for this (date, symbol) to ensure strict idempotency
            con.execute("DELETE FROM daily_prices WHERE date = ? AND symbol = ?;", [rec["date"], rec["symbol"]])
            con.execute("""
                INSERT INTO daily_prices (date, symbol, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """, [rec["date"], rec["symbol"], rec["open"], rec["high"], rec["low"], rec["close"], rec["volume"]])

        total_rows = con.execute("SELECT count(*) FROM daily_prices;").fetchone()[0]
        distinct_dates = con.execute("SELECT count(distinct date) FROM daily_prices;").fetchone()[0]
        logger.info(f"DuckDB sync complete. Total rows: {total_rows} across {distinct_dates} sessions.")

        # Export to Parquet
        logger.info(f"Exporting updated archive to {PARQUET_PATH}...")
        all_df = con.execute("SELECT * FROM daily_prices ORDER BY date, symbol;").fetchdf()
        all_df.to_parquet(PARQUET_PATH, index=False)
        parquet_size_kb = os.path.getsize(PARQUET_PATH) / 1024.0
        logger.info(f"✅ Parquet archive successfully exported ({parquet_size_kb:.1f} KB).")

    finally:
        con.close()

    logger.info("EOD Archival Pipeline completed successfully.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APEX QUANT 999 - EOD Archival Job")
    parser.add_argument("--force", action="store_true", help="Bypass 15:45 IST and weekend/holiday timing check")
    parser.add_argument("--date", type=str, default=None, help="Specific target date (YYYY-MM-DD)")
    args = parser.parse_args()

    success = run_eod_pipeline(force=args.force, target_date_str=args.date)
    sys.exit(0 if success else 1)
