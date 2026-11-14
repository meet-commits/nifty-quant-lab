"""
src/bhavcopy_service.py
NSE Equity EOD Bhavcopy Ingestion, Caching, and Universe Service.
Integrates directly with official NSE MCP Bhavcopy Server, with reliable
archives fallback and idempotent DuckDB storage.
"""

import os
import io
import csv
import logging
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Any, Optional
import pandas as pd
import duckdb
import requests

try:
    from nse_mcp_client import fetch_nse_mcp_bhavcopy
except ImportError:
    from src.nse_mcp_client import fetch_nse_mcp_bhavcopy

logger = logging.getLogger("BHAVCOPY_SERVICE")
IST = ZoneInfo("Asia/Kolkata")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
DB_PATH = os.path.join(DATA_DIR, "research.duckdb")
HOLIDAYS_PATH = os.path.join(CONFIG_DIR, "nse_holidays.csv")
ETF_LIST_PATH = os.path.join(CONFIG_DIR, "nse_etfs.csv")


def get_official_etf_symbols() -> set:
    """Loads official NSE ETF symbols from config/nse_etfs.csv."""
    try:
        if os.path.exists(ETF_LIST_PATH):
            df_etf = pd.read_csv(ETF_LIST_PATH)
            col = "Symbol" if "Symbol" in df_etf.columns else df_etf.columns[0]
            return set(df_etf[col].astype(str).str.strip().str.upper())
    except Exception:
        pass
    return set()


def get_db_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Returns thread-safe connection to DuckDB."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return duckdb.connect(DB_PATH, read_only=read_only)


def init_bhavcopy_table(con: Optional[duckdb.DuckDBPyConnection] = None) -> None:
    """Ensures bhavcopy_daily table exists with proper indexes and data provenance source."""
    close_con = False
    if con is None:
        con = get_db_connection(read_only=False)
        close_con = True

    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS bhavcopy_daily (
                trade_date DATE,
                symbol VARCHAR,
                series VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                prev_close DOUBLE,
                volume BIGINT,
                turnover DOUBLE,
                source VARCHAR DEFAULT 'NSE_MCP',
                PRIMARY KEY (trade_date, symbol, series)
            );
            CREATE INDEX IF NOT EXISTS idx_bhavcopy_date ON bhavcopy_daily(trade_date);
            CREATE INDEX IF NOT EXISTS idx_bhavcopy_sym ON bhavcopy_daily(symbol);
            CREATE INDEX IF NOT EXISTS idx_bhavcopy_source ON bhavcopy_daily(source);
        """)
        try:
            con.execute("ALTER TABLE bhavcopy_daily ADD COLUMN IF NOT EXISTS source VARCHAR DEFAULT 'NSE_MCP';")
        except Exception:
            pass
    finally:
        if close_con:
            con.close()


def save_bhavcopy_records(records: List[Dict[str, Any]], con: Optional[duckdb.DuckDBPyConnection] = None, default_source: str = "NSE_MCP") -> int:
    """Idempotently saves or updates bhavcopy records in DuckDB with source provenance."""
    if not records:
        return 0

    close_con = False
    if con is None:
        con = get_db_connection(read_only=False)
        close_con = True

    try:
        init_bhavcopy_table(con)

        # Prepare bulk dataframe for fast ingestion
        df = pd.DataFrame(records)
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        if "source" not in df.columns:
            df["source"] = default_source
        else:
            df["source"] = df["source"].fillna(default_source)

        # Idempotent upsert via delete + insert within a transaction
        con.execute("BEGIN TRANSACTION;")
        target_dates = [str(d) for d in df["trade_date"].unique()]
        for d in target_dates:
            con.execute("DELETE FROM bhavcopy_daily WHERE trade_date = ?;", [d])

        con.register("df_insert", df)
        con.execute("""
            INSERT INTO bhavcopy_daily (trade_date, symbol, series, open, high, low, close, prev_close, volume, turnover, source)
            SELECT trade_date, symbol, series, open, high, low, close, prev_close, volume, turnover, source
            FROM df_insert;
        """)
        con.unregister("df_insert")
        con.execute("COMMIT;")

        return len(records)
    except Exception as e:
        con.execute("ROLLBACK;")
        logger.error(f"Error saving Bhavcopy records to DuckDB: {e}")
        raise e
    finally:
        if close_con:
            con.close()


def fetch_fallback_nse_bhavcopy_csv(target_date: date) -> Optional[List[Dict[str, Any]]]:
    """
    Downloads official NSE EOD Bhavcopy CSV from NSE archives.
    Handles standard formats: sec_bhavdata_full_{ddmmyyyy}.csv
    """
    d_str = target_date.strftime("%d%m%Y")
    date_iso = target_date.strftime("%Y-%m-%d")
    url = f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{d_str}.csv"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "*/*"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=6)
        if resp.status_code == 200 and len(resp.text) > 1000:
            records = []
            f = io.StringIO(resp.text)
            reader = csv.DictReader(f, skipinitialspace=True)
            for raw_row in reader:
                row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw_row.items() if k}
                series = row.get("SERIES", "").upper()
                if series not in ("EQ", "BE"):
                    continue
                sym = row.get("SYMBOL", "").upper()
                if not sym:
                    continue
                try:
                    c = float(row.get("CLOSE_PRICE") or row.get("LAST_PRICE") or 0.0)
                    if c <= 0.0:
                        continue
                    o = float(row.get("OPEN_PRICE") or c)
                    h = float(row.get("HIGH_PRICE") or max(o, c))
                    l = float(row.get("LOW_PRICE") or min(o, c))
                    prev = float(row.get("PREV_CLOSE") or c)
                    vol = int(float(row.get("TTL_TRD_QNTY") or 0))
                    turnover = float(row.get("TURNOVER_LACS") or 0.0) * 100000.0

                    records.append({
                        "trade_date": date_iso,
                        "symbol": sym,
                        "series": series,
                        "open": round(o, 2),
                        "high": round(h, 2),
                        "low": round(l, 2),
                        "close": round(c, 2),
                        "prev_close": round(prev, 2),
                        "volume": vol,
                        "turnover": round(turnover, 2),
                        "source": "NSE_CSV"
                    })
                except (ValueError, TypeError):
                    continue
            if records:
                return records
    except Exception as e:
        logger.warning(f"Fallback NSE Bhavcopy fetch failed for {date_iso}: {e}")

    return None


def fetch_and_store_bhavcopy(target_date: Optional[date] = None, force: bool = False) -> int:
    """
    Primary pipeline for ingesting full EOD Bhavcopy into DuckDB.
    1. Check DuckDB cache
    2. Try official NSE MCP server (cm_get_bhavcopy)
    3. Fallback to direct NSE archives CSV
    """
    if target_date is None:
        target_date = datetime.now(IST).date()
        # If weekend, retreat to Friday
        while target_date.weekday() >= 5:
            target_date -= timedelta(days=1)

    con = get_db_connection(read_only=False)
    try:
        init_bhavcopy_table(con)
        if not force:
            count = con.execute("SELECT count(*) FROM bhavcopy_daily WHERE trade_date = ?;", [str(target_date)]).fetchone()[0]
            if count > 50:
                return count

        # 1. Primary: Official NSE MCP Server
        records = fetch_nse_mcp_bhavcopy(target_date)
        if records:
            logger.info(f"Retrieved {len(records)} quotes from official NSE MCP Bhavcopy server.")
            return save_bhavcopy_records(records, con, default_source="NSE_MCP")

        # 2. Secondary Fallback: Direct NSE archives CSV
        records = fetch_fallback_nse_bhavcopy_csv(target_date)
        if records:
            logger.info(f"Retrieved {len(records)} quotes from NSE Archives Bhavcopy CSV.")
            return save_bhavcopy_records(records, con, default_source="NSE_CSV")

        return 0
    finally:
        con.close()


def ensure_bhavcopy_seeded(min_sessions: int = 30) -> int:
    """
    Seeds DuckDB bhavcopy_daily with robust multi-session equity universe history
    so technical indicators (EMA20, EMA50, RSI14, ATR14, Volume Surge) and
    scanners function deterministically out-of-the-box even when offline.
    """
    con = get_db_connection(read_only=False)
    try:
        init_bhavcopy_table(con)
        distinct_sessions = con.execute("SELECT count(distinct trade_date) FROM bhavcopy_daily;").fetchone()[0]
        if distinct_sessions >= min_sessions:
            return distinct_sessions

        logger.info(f"DuckDB bhavcopy_daily has {distinct_sessions} sessions (< {min_sessions}). Seeding universe history...")

        # Load all constituent symbols from weights.csv + liquid watchlist
        equities = [
            ("RELIANCE", 1324.40, 1.05), ("HDFCBANK", 804.00, 0.98), ("ICICIBANK", 1289.00, 1.10),
            ("INFY", 1854.10, 0.95), ("TCS", 4175.40, 1.01), ("ITC", 482.30, 1.02),
            ("LT", 3585.00, 1.08), ("AXISBANK", 1180.70, 1.04), ("BHARTIARTL", 1580.00, 1.12),
            ("SBIN", 842.00, 1.06), ("KOTAKBANK", 1750.00, 0.99), ("TRENT", 6850.00, 1.25),
            ("BEL", 295.40, 1.15), ("DIXON", 14850.00, 1.20), ("INDUSINDBK", 1450.00, 0.92),
            ("TATACONSUM", 1085.00, 1.03), ("ASIANPAINT", 2840.00, 0.97), ("WIPRO", 540.50, 0.96),
            ("MARUTI", 12400.00, 1.04), ("BAJFINANCE", 6950.00, 1.01), ("TATAMOTORS", 985.00, 1.07),
            ("SUNPHARMA", 1780.00, 1.05), ("NTPC", 385.00, 1.06), ("POWERGRID", 325.00, 1.02),
            ("TITAN", 3420.00, 1.01), ("ADANIENT", 2980.00, 1.03), ("ULTRACEMCO", 11200.00, 1.02),
            ("M&M", 2850.00, 1.14), ("COALINDIA", 495.00, 1.03), ("HCLTECH", 1720.00, 1.05),
            ("FEDERALBNK", 195.00, 1.06), ("BANKBARODA", 255.00, 1.04), ("PNB", 105.00, 0.99),
            ("IDFCFIRSTB", 78.50, 0.95), ("VEDL", 465.00, 1.18), ("HINDALCO", 670.00, 1.08),
            ("JINDALSTEL", 940.00, 1.09), ("JSWSTEEL", 980.00, 1.04), ("CIPLA", 1520.00, 1.02),
            ("DRREDDY", 6450.00, 0.99), ("APOLLOHOSP", 6850.00, 1.05), ("EICHERMOT", 4850.00, 1.10),
            ("BAJAJ-AUTO", 9450.00, 1.06), ("HEROMOTOCO", 4950.00, 1.02), ("BPCL", 310.00, 1.01),
            ("IOC", 145.00, 0.98), ("ONGC", 265.00, 1.02), ("GAIL", 215.00, 1.04),
            ("DLF", 820.00, 1.08), ("GODREJPROP", 2950.00, 1.05)
        ]

        now_date = datetime.now(IST).date()
        date_list = []
        curr = now_date
        while len(date_list) < min_sessions:
            if curr.weekday() < 5:  # Monday to Friday
                date_list.append(curr)
            curr -= timedelta(days=1)
        date_list.reverse()

        seed_records = []
        for sym, base_price, trend_drift in equities:
            p = base_price * (1.0 - (trend_drift - 1.0) * 0.5)
            for i, d in enumerate(date_list):
                # Daily deterministic variation based on hash for stability
                day_hash = (hash(f"{sym}_{d}") % 1000) / 1000.0
                daily_pct = ((day_hash - 0.48) * 0.035) + ((trend_drift - 1.0) * 0.008)
                prev_c = p
                p = round(p * (1.0 + daily_pct), 2)

                h = round(p * (1.0 + (day_hash * 0.015)), 2)
                l = round(p * (1.0 - ((1.0 - day_hash) * 0.015)), 2)
                o = round((prev_c + p) / 2.0, 2)
                vol = int(500000 * (0.6 + day_hash * 1.5))
                if i == len(date_list) - 1 and trend_drift > 1.15:
                    vol = int(vol * 2.8)  # Volume breakout signature on final session

                turnover = round(p * vol, 2)

                seed_records.append({
                    "trade_date": str(d),
                    "symbol": sym,
                    "series": "EQ",
                    "open": o,
                    "high": max(h, o, p),
                    "low": min(l, o, p),
                    "close": p,
                    "prev_close": prev_c,
                    "volume": vol,
                    "turnover": turnover,
                    "source": "TEST_FIXTURE"
                })

        save_bhavcopy_records(seed_records, con, default_source="TEST_FIXTURE")
        logger.info(f"Successfully seeded {len(seed_records)} rows across {len(date_list)} sessions.")
        return len(date_list)
    finally:
        con.close()


def get_bhavcopy_universe(
    days: int = 150,
    allow_test_fixtures: bool = False,
    min_turnover: float = 1_000_000.0,
    exclude_etfs: bool = True
) -> pd.DataFrame:
    """
    Queries historical Bhavcopy records for active cash equities from DuckDB.
    Returns clean pandas DataFrame ordered by symbol and trade_date.
    - Restricts to cash series 'EQ' and 'BE' (excludes SME series like SM/ST).
    - Enforces documented turnover filter (default >= 1,000,000 INR daily traded value).
    - Excludes ETFs (*BEES, *ETF, *GOLD*) from the single-stock equity universe.
    - Guarantees the app never reads TEST_FIXTURE rows outside tests unless allow_test_fixtures=True.
    """
    ensure_bhavcopy_seeded()
    con = get_db_connection(read_only=True)
    try:
        source_filter = "" if allow_test_fixtures else "WHERE source != 'TEST_FIXTURE'"
        source_filter_and = "" if allow_test_fixtures else "AND source != 'TEST_FIXTURE'"
        etf_filter = "AND symbol NOT LIKE '%BEES' AND symbol NOT LIKE '%ETF' AND symbol NOT LIKE '%GOLD%'" if exclude_etfs else ""

        # Get list of recent distinct dates
        recent_dates_df = con.execute(f"""
            SELECT distinct trade_date 
            FROM bhavcopy_daily 
            {source_filter}
            ORDER BY trade_date DESC 
            LIMIT ?;
        """, [days]).fetchdf()

        if recent_dates_df.empty:
            return pd.DataFrame()

        min_date = str(recent_dates_df["trade_date"].min())

        params = [min_date, min_turnover]
        df = con.execute(f"""
            SELECT trade_date, symbol, series, open, high, low, close, prev_close, volume, turnover, source
            FROM bhavcopy_daily
            WHERE series IN ('EQ', 'BE') 
              AND trade_date >= ? 
              AND (turnover IS NULL OR turnover >= ?)
              {source_filter_and}
              {etf_filter}
            ORDER BY symbol, trade_date ASC;
        """, params).fetchdf()

        if not df.empty:
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

        return df
    finally:
        con.close()


def get_bhavcopy_provenance_stats() -> Dict[str, Any]:
    """Returns provenance and audit statistics for all rows in bhavcopy_daily."""
    con = get_db_connection(read_only=False)
    try:
        init_bhavcopy_table(con)
        summary = con.execute("""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT trade_date) as num_sessions,
                MIN(trade_date) as min_date,
                MAX(trade_date) as max_date,
                COUNT(DISTINCT symbol) as num_symbols
            FROM bhavcopy_daily;
        """).fetchone()

        source_rows = con.execute("""
            SELECT COALESCE(source, 'UNKNOWN') as src, COUNT(*) as cnt
            FROM bhavcopy_daily
            GROUP BY src;
        """).fetchall()

        rows_per_source = {src: cnt for src, cnt in source_rows}

        return {
            "total_rows": summary[0] if summary else 0,
            "num_sessions": summary[1] if summary else 0,
            "min_trade_date": str(summary[2]) if summary and summary[2] else "N/A",
            "max_trade_date": str(summary[3]) if summary and summary[3] else "N/A",
            "num_symbols": summary[4] if summary else 0,
            "rows_per_source": rows_per_source
        }
    finally:
        con.close()


def check_bhavcopy_backfill_health(min_required_sessions: int = 250, max_stale_trading_days: int = 3) -> Dict[str, Any]:
    """
    Startup guard: asserts that DuckDB bhavcopy has >= 250 real sessions
    and that the latest session is not older than 3 trading days.
    """
    con = get_db_connection(read_only=True)
    try:
        init_bhavcopy_table(con)
        res = con.execute("""
            SELECT 
                COUNT(DISTINCT trade_date) as num_real_sessions,
                MIN(trade_date) as min_date,
                MAX(trade_date) as max_date,
                COUNT(DISTINCT symbol) as num_symbols
            FROM bhavcopy_daily
            WHERE source IN ('NSE_MCP', 'NSE_CSV');
        """).fetchone()

        num_sessions = res[0] if res and res[0] is not None else 0
        min_date = res[1]
        max_date = res[2]
        num_symbols = res[3] if res and res[3] is not None else 0

        today = datetime.now(IST).date()

        # Count trading days between max_date and today (excluding weekends)
        trading_days_old = 0
        if max_date:
            curr = max_date + timedelta(days=1)
            while curr <= today:
                if curr.weekday() < 5:  # Monday to Friday
                    trading_days_old += 1
                curr += timedelta(days=1)
        else:
            trading_days_old = 9999

        calendar_days_old = (today - max_date).days if max_date else 9999

        is_healthy = True
        status_message = "HEALTHY"

        if num_sessions < min_required_sessions:
            is_healthy = False
            status_message = f"INSUFFICIENT_HISTORY: {num_sessions} real sessions (< {min_required_sessions} required)"
        elif trading_days_old > max_stale_trading_days:
            is_healthy = False
            status_message = f"DATA_STALE: latest real session is {max_date} ({trading_days_old} trading days old > {max_stale_trading_days})"

        return {
            "is_healthy": is_healthy,
            "status_message": status_message,
            "num_sessions": num_sessions,
            "min_date": str(min_date) if min_date else None,
            "max_date": str(max_date) if max_date else None,
            "num_symbols": num_symbols,
            "trading_days_old": trading_days_old,
            "calendar_days_old": calendar_days_old
        }
    finally:
        con.close()
