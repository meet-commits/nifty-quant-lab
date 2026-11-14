"""
scripts/backfill_bhavcopy.py
Idempotent and resumable backfill of >= 300 real NSE EQ/BE trading sessions
into DuckDB bhavcopy_daily via official NSE archives / MCP client.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo
import pandas as pd

# Add repo root to pythonpath
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from src.bhavcopy_service import (
    get_db_connection,
    init_bhavcopy_table,
    fetch_fallback_nse_bhavcopy_csv,
    save_bhavcopy_records,
    DB_PATH
)

IST = ZoneInfo("Asia/Kolkata")

def run_backfill(target_session_count: int = 305, max_workers: int = 6):
    print("=" * 80)
    print(f"🚀 INITIATING REAL BHAVCOPY BACKFILL (Target: >= {target_session_count} Sessions)")
    print(f"DuckDB Target: {DB_PATH}")
    print("=" * 80)

    # 1. Connect and initialize table
    con = get_db_connection(read_only=False)
    init_bhavcopy_table(con)

    # 2. Get existing completed real dates
    existing_real_dates = set(
        r[0] for r in con.execute("""
            SELECT DISTINCT trade_date 
            FROM bhavcopy_daily 
            WHERE source IN ('NSE_MCP', 'NSE_CSV');
        """).fetchall()
    )
    con.close()

    print(f"📊 Currently backfilled real sessions: {len(existing_real_dates)}")

    # 3. Determine target dates from market_data.parquet
    parquet_path = os.path.join(BASE_DIR, "data", "market_data.parquet")
    df_mkt = pd.read_parquet(parquet_path)
    nifty = df_mkt[df_mkt["symbol"] == "NIFTY_SPOT"].sort_values("date")
    trading_dates = [d.date() if hasattr(d, "date") else d for d in pd.to_datetime(nifty["date"]).tolist()]
    
    # Take the latest target_session_count trading dates
    target_dates = trading_dates[-target_session_count:]
    missing_dates = [d for d in target_dates if d not in existing_real_dates]

    print(f"🎯 Target range: {target_dates[0]} to {target_dates[-1]} ({len(target_dates)} sessions)")
    print(f"⏳ Missing sessions to download: {len(missing_dates)}")

    if not missing_dates:
        print("✅ Backfill already complete. Verifying statistics...")
        print_audit_stats()
        return

    # 4. Concurrently download and batch-save
    print(f"⚡ Downloading {len(missing_dates)} sessions with {max_workers} worker threads...")
    t0 = time.time()
    downloaded_count = 0
    saved_rows = 0

    # Batch download in chunks of 20 to insert progressively
    chunk_size = 15
    for i in range(0, len(missing_dates), chunk_size):
        chunk = missing_dates[i:i + chunk_size]
        print(f"\nFetching chunk {i//chunk_size + 1}/{(len(missing_dates)-1)//chunk_size + 1} ({len(chunk)} dates: {chunk[0]} to {chunk[-1]})...")
        
        batch_records = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_date = {executor.submit(fetch_fallback_nse_bhavcopy_csv, d): d for d in chunk}
            for future in as_completed(future_to_date):
                d = future_to_date[future]
                try:
                    recs = future.result()
                    if recs:
                        batch_records.extend(recs)
                        downloaded_count += 1
                        print(f"  ✓ {d}: {len(recs)} equities")
                    else:
                        print(f"  ✗ {d}: No records found (possible trading holiday)")
                except Exception as ex:
                    print(f"  ✗ {d}: Error: {ex}")

        if batch_records:
            con_write = get_db_connection(read_only=False)
            try:
                n = save_bhavcopy_records(batch_records, con_write, default_source="NSE_CSV")
                saved_rows += n
                print(f"  💾 Saved {n} records to DuckDB.")
            finally:
                con_write.close()

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"🎉 BACKFILL COMPLETE in {elapsed:.2f}s! ({downloaded_count} sessions downloaded, {saved_rows:,} rows saved)")
    print("=" * 80)

    print_audit_stats()


def print_audit_stats():
    con = get_db_connection(read_only=True)
    try:
        # 1. Sessions and date range
        stats = con.execute("""
            SELECT 
                COUNT(DISTINCT trade_date) as sessions,
                MIN(trade_date) as min_date,
                MAX(trade_date) as max_date,
                COUNT(DISTINCT symbol) as unique_symbols,
                COUNT(*) as total_rows
            FROM bhavcopy_daily
            WHERE source IN ('NSE_MCP', 'NSE_CSV');
        """).fetchone()

        sessions, min_date, max_date, symbols, total_rows = stats

        # 2. Rows per source
        source_counts = con.execute("""
            SELECT source, COUNT(*), COUNT(DISTINCT trade_date)
            FROM bhavcopy_daily
            GROUP BY source
            ORDER BY source;
        """).fetchall()

        # 3. Symbols eligible for 200 EMA (having >= 200 sessions)
        eligible_200 = con.execute("""
            WITH sym_counts AS (
                SELECT symbol, COUNT(DISTINCT trade_date) as cnt
                FROM bhavcopy_daily
                WHERE source IN ('NSE_MCP', 'NSE_CSV')
                GROUP BY symbol
            )
            SELECT COUNT(*) FROM sym_counts WHERE cnt >= 200;
        """).fetchone()[0]

        print("\n--- RAW AUDIT EVIDENCE: BHAVCOPY PROVENANCE & HEALTH ---")
        print(f"Real Trading Sessions : {sessions}")
        print(f"Date Range            : {min_date} to {max_date}")
        print(f"Distinct Symbols      : {symbols}")
        print(f"Total Real Rows       : {total_rows:,}")
        print(f"Eligible for 200 EMA  : {eligible_200} symbols (>= 200 real sessions)")
        print("\nRows Per Source:")
        for src, cnt, s_sess in source_counts:
            print(f"  - {src:15s}: {cnt:>10,d} rows across {s_sess:>3d} sessions")
        print("--------------------------------------------------------\n")
    finally:
        con.close()


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 305
    run_backfill(target_session_count=count)
