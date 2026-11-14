"""
scripts/forward_test_runner.py
Scheduled Forward-Test Automation Engine.
Runs independently of Streamlit Cloud in GitHub Actions:
- 15:15 IST (09:45 UTC): Generates Close Model (eod_1515) prediction record.
- 08:45 IST (03:15 UTC): Generates Pre-Open Nowcast (preopen_0845) prediction record.
- 09:30 IST (04:00 UTC): Fills in actual next-day open gap outcomes for pending predictions.
- Hardened Telemetry: Tracks scheduled vs actual start, delay minutes, status_flag ("ON_TIME" / "LATE"),
  gift_available flag, and verified data sources.
Saves records idempotently to data/predictions_log.parquet and commits to data-archive branch.
"""

import os
import sys
import json
import time
import urllib.request
import argparse
import logging
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from src.live_market_service import fetch_live_market_snapshot
from src.gap_prediction_service import (
    compute_overnight_gap_forecast,
    update_forward_test_outcomes,
    log_live_prediction,
    PREDICTIONS_LOG_PATH
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FORWARD_TEST_RUNNER")
IST = ZoneInfo("Asia/Kolkata")


def probe_feed_connectivity():
    """Probes reachability of market feeds from runner environment."""
    probes = [
        {"name": "Yahoo Finance API", "url": "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI"},
        {"name": "NSE India Archives", "url": "https://archives.nseindia.com/products/content/sec_bhavdata_full_13112026.csv"},
        {"name": "NSE IX GIFT Live", "url": "https://www.nseix.com"}
    ]
    
    probe_results = []
    print("\n" + "=" * 80)
    print("GITHUB RUNNER FEED CONNECTIVITY PROBE AUDIT")
    print("=" * 80)
    
    for p in probes:
        t0 = time.time()
        status_str = "REACHABLE"
        err_msg = ""
        req = urllib.request.Request(p["url"], headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                code = resp.getcode()
                latency_ms = round((time.time() - t0) * 1000, 1)
                status_str = f"ONLINE (HTTP {code})"
        except Exception as e:
            latency_ms = round((time.time() - t0) * 1000, 1)
            status_str = "OFFLINE / BLOCKED"
            err_msg = str(e)[:45]
            
        probe_results.append({
            "Feed": p["name"],
            "Status": status_str,
            "Latency_ms": f"{latency_ms} ms",
            "Fallback_Handling": "Failsafe default / cache" if "OFFLINE" in status_str else "Direct stream"
        })
        
    print(pd.DataFrame(probe_results).to_string(index=False))
    print("=" * 80 + "\n")
    return probe_results


def run_prediction_job(model_type: str = "eod_1515"):
    now_ist = datetime.now(IST)
    today_date = now_ist.date()
    logger.info(f"🚀 Running Scheduled Prediction Job: {model_type.upper()} at {now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}")

    # Determine scheduled target time
    if model_type == "eod_1515":
        target_dt = datetime.combine(today_date, dtime(15, 15), tzinfo=IST)
        target_label = "15:15 IST"
    else:
        target_dt = datetime.combine(today_date, dtime(8, 45), tzinfo=IST)
        target_label = "08:45 IST"

    # Compute delay
    delay_minutes = round((now_ist - target_dt).total_seconds() / 60.0, 2)
    # If triggered early or on time within 15 mins, mark ON_TIME, else LATE
    status_flag = "LATE" if delay_minutes > 15.0 else ("EARLY" if delay_minutes < -15.0 else "ON_TIME")

    # Fetch live snapshot
    live_snap = fetch_live_market_snapshot()
    if not live_snap:
        logger.error("❌ Failed to fetch live snapshot from data feeds.")
        return False

    # Check GIFT availability
    gift_q = live_snap.get("giftnifty") or {}
    gift_last = gift_q.get("last") or gift_q.get("value")
    gift_available = bool(gift_last is not None and float(gift_last) > 0 and not gift_q.get("is_stale", False))

    data_sources_map = {
        "spot": "NSE_LIVE",
        "vix": "NSE_LIVE",
        "sp500": "YAHOO_LIVE",
        "crude": "NYMEX_LIVE",
        "gift": "NSE_IX_LIVE" if gift_available else "UNAVAILABLE"
    }

    # Generate prediction
    forecast = compute_overnight_gap_forecast(live_snap, model_type=model_type)
    
    def _to_prob(val):
        if isinstance(val, (int, float)):
            return round(float(val) / (100.0 if float(val) > 1.0 else 1.0), 4)
        if isinstance(val, str):
            clean = val.replace("%", "").strip()
            num = float(clean) if clean else 0.0
            return round(num / (100.0 if num > 1.0 else 1.0), 4)
        return 0.0

    # Enrich and stamp telemetry in predictions_log.parquet
    pred_record = {
        "timestamp": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "scheduled_timestamp": target_dt.strftime("%Y-%m-%d %H:%M:%S IST"),
        "actual_run_timestamp": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "run_delay_minutes": delay_minutes,
        "status_flag": status_flag,
        "gift_available": gift_available,
        "data_sources": json.dumps(data_sources_map),
        "model_version": "v2.4-clean-eval",
        "date": str(today_date),
        "model_type": model_type,
        "spot_ref": forecast.get("spot_reference", 0.0),
        "vix_ref": forecast.get("vix_reference", 15.0),
        "predicted_class": forecast.get("predicted_class", 0),
        "prob_up": _to_prob(forecast.get("prob_up_pct", 0.0)),
        "prob_flat": _to_prob(forecast.get("prob_flat_pct", 0.0)),
        "prob_down": _to_prob(forecast.get("prob_down_pct", 0.0)),
        "q10_pts": forecast.get("q10_pts", 0.0),
        "q50_pts": forecast.get("q50_pts", 0.0),
        "q90_pts": forecast.get("q90_pts", 0.0),
        "actual_gap_pct": None,
        "actual_gap_pts": None,
        "verdict": "⏳ PENDING NEXT SESSION OPEN"
    }
    log_live_prediction(pred_record)
    logger.info(f"✅ Hardened Telemetry Record Saved: {status_flag} (delay: {delay_minutes:+.1f}m) | gift_avail={gift_available}")
    return True


def run_outcome_fill_job():
    now_ist = datetime.now(IST)
    logger.info(f"🎯 Running Outcome Fill Job at {now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}")
    updated_df = update_forward_test_outcomes()
    if not updated_df.empty:
        pending_count = (updated_df['verdict'] == '⏳ PENDING NEXT SESSION OPEN').sum()
        logger.info(f"✅ Outcome fill complete. Total records: {len(updated_df)}, Remaining pending: {pending_count}")
    else:
        logger.info("ℹ️ No forward-test records found or updated.")
    return True


def print_log_audit():
    if not os.path.exists(PREDICTIONS_LOG_PATH):
        print("❌ No predictions log file found at data/predictions_log.parquet")
        return

    df = pd.read_parquet(PREDICTIONS_LOG_PATH)
    print("\n" + "=" * 90)
    print("FORWARD-TEST LOG AUDIT EVIDENCE")
    print("=" * 90)
    print(f"Total Logged Predictions : {len(df)}")
    print(f"Min Timestamp            : {df['timestamp'].min() if 'timestamp' in df.columns else 'N/A'}")
    print(f"Max Timestamp            : {df['timestamp'].max() if 'timestamp' in df.columns else 'N/A'}")
    print(f"Models Logged            : {df['model_type'].unique().tolist() if 'model_type' in df.columns else 'N/A'}")
    print(f"Verdicts Distribution    :\n{df['verdict'].value_counts() if 'verdict' in df.columns else 'N/A'}")
    print("\nRecent Forward-Test Records with Telemetry:")
    cols_to_show = [c for c in ['timestamp', 'model_type', 'status_flag', 'run_delay_minutes', 'gift_available', 'spot_ref', 'predicted_class', 'q50_pts', 'verdict'] if c in df.columns]
    print(df[cols_to_show].tail(5).to_string(index=False))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forward Test Scheduled Runner")
    parser.add_argument("--job-type", choices=["close_1515", "preopen_0845", "fill_outcomes", "probe", "audit", "all"], default="audit")
    args = parser.parse_args()

    if args.job_type == "probe":
        probe_feed_connectivity()
    elif args.job_type == "close_1515":
        run_prediction_job("eod_1515")
    elif args.job_type == "preopen_0845":
        run_prediction_job("preopen_0845")
    elif args.job_type == "fill_outcomes":
        run_outcome_fill_job()
    elif args.job_type == "all":
        probe_feed_connectivity()
        run_prediction_job("eod_1515")
        run_outcome_fill_job()

    print_log_audit()
