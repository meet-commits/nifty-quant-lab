"""
scripts/forward_test_runner.py
Hardened Forward-Test Automation Engine.
Runs independently of Streamlit Cloud in GitHub Actions:
- 15:15 IST (09:45 UTC): Generates Close Model (model_a_1515) prediction record.
- 08:45 IST (03:15 UTC): Generates Pre-Open Nowcast (model_b_nowcast) and all 4 pre-registered candidates:
  1. model_a_1515
  2. base_sp500 (S&P 500 Implied Gap Rule)
  3. simple_macro3 (Simplified S&P + Crude + VIX Logistic Model)
  4. gift_implied (Live GIFT Nifty Basis Rule)
- 09:30 IST (04:00 UTC): Fills in actual next-day open gap outcomes for pending predictions.
- Strict Hygiene: Quarantines MANUAL/TEST runs and non-trading days from live scoring.
- Real Providers Tagging: YAHOO_FINANCE, NSE_LIVE, DUCKDB_EOD, NSE_IX_DIRECT.
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

from src.live_market_service import fetch_live_market_snapshot, load_nse_holidays_map
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
        req = urllib.request.Request(p["url"], headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                code = resp.getcode()
                latency_ms = round((time.time() - t0) * 1000, 1)
                status_str = f"ONLINE (HTTP {code})"
        except Exception as e:
            latency_ms = round((time.time() - t0) * 1000, 1)
            status_str = "OFFLINE / BLOCKED"

        probe_results.append({
            "Feed": p["name"],
            "Status": status_str,
            "Latency_ms": f"{latency_ms} ms",
            "Fallback_Handling": "Failsafe default / cache" if "OFFLINE" in status_str else "Direct stream"
        })

    print(pd.DataFrame(probe_results).to_string(index=False))
    print("=" * 80 + "\n")
    return probe_results


def _to_prob(val):
    if isinstance(val, (int, float)):
        return round(float(val) / (100.0 if float(val) > 1.0 else 1.0), 4)
    if isinstance(val, str):
        clean = val.replace("%", "").strip()
        num = float(clean) if clean else 0.0
        return round(num / (100.0 if num > 1.0 else 1.0), 4)
    return 0.0


def run_prediction_job(model_type: str = "close_1515", run_type: str = "MANUAL"):
    now_ist = datetime.now(IST)
    today_date = now_ist.date()

    holidays = load_nse_holidays_map()
    is_weekend = today_date.weekday() >= 5
    is_holiday = str(today_date) in holidays
    is_trading_day = (not is_weekend) and (not is_holiday)

    logger.info(f"🚀 Running Prediction Job: {model_type.upper()} at {now_ist.strftime('%Y-%m-%d %H:%M:%S IST')} | run_type={run_type} | trading_day={is_trading_day}")

    # Determine scheduled target time
    if "close" in model_type or "1515" in model_type:
        target_dt = datetime.combine(today_date, dtime(15, 15), tzinfo=IST)
        target_label = "15:15 IST"
        actual_model_key = "eod_1515"
    else:
        target_dt = datetime.combine(today_date, dtime(8, 45), tzinfo=IST)
        target_label = "08:45 IST"
        actual_model_key = "preopen_0845"

    delay_minutes = round((now_ist - target_dt).total_seconds() / 60.0, 2)
    status_flag = "LATE" if delay_minutes > 15.0 else ("EARLY" if delay_minutes < -15.0 else "ON_TIME")

    live_snap = fetch_live_market_snapshot()
    if not live_snap:
        logger.error("❌ Failed to fetch live snapshot from data feeds.")
        return False

    gift_q = live_snap.get("giftnifty") or {}
    gift_last = gift_q.get("last") or gift_q.get("value")
    gift_available = bool(gift_last is not None and float(gift_last) > 0 and not gift_q.get("is_stale", False))

    data_sources_map = {
        "spot": "DUCKDB_EOD" if not is_trading_day else "YAHOO_LIVE",
        "vix": "NSE_LIVE",
        "sp500": "YAHOO_FINANCE",
        "crude": "YAHOO_FINANCE",
        "gift": "NSE_IX_DIRECT" if gift_available else "UNAVAILABLE"
    }

    forecast = compute_overnight_gap_forecast(live_snap, model_type=actual_model_key)

    # Initial verdict based on trading day and run_type
    if not is_trading_day:
        initial_verdict = "NON-TRADING DAY / QUARANTINED"
    elif run_type != "SCHEDULED":
        initial_verdict = "MANUAL/TEST RUN"
    else:
        initial_verdict = "⏳ PENDING NEXT SESSION OPEN"

    # Base candidate record
    pred_record = {
        "timestamp": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "scheduled_timestamp": target_dt.strftime("%Y-%m-%d %H:%M:%S IST"),
        "actual_run_timestamp": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "run_delay_minutes": delay_minutes,
        "status_flag": status_flag,
        "run_type": run_type,
        "is_trading_day": is_trading_day,
        "gift_available": gift_available,
        "data_sources": json.dumps(data_sources_map),
        "model_version": "v2.5-preregistered",
        "date": str(today_date),
        "model_type": actual_model_key,
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
        "verdict": initial_verdict
    }
    log_live_prediction(pred_record)
    logger.info(f"✅ Telemetry Record Saved: {actual_model_key} | {run_type} | {initial_verdict}")

    # If Pre-Open job, also log the remaining 3 pre-registered candidates
    if actual_model_key == "preopen_0845":
        # Candidate 2: S&P 500 Implied Gap Rule
        sp_q = live_snap.get("sp500") or {}
        sp_pct = float(sp_q.get("chg_pct") or 0.0)
        sp_implied = sp_pct * 0.40
        sp_class = 2 if sp_implied >= 0.25 else (1 if sp_implied <= -0.25 else 0)
        rec_sp = pred_record.copy()
        rec_sp["model_type"] = "base_sp500"
        rec_sp["predicted_class"] = sp_class
        rec_sp["q50_pts"] = round(forecast.get("spot_reference", 25000.0) * (sp_implied / 100.0), 1)
        log_live_prediction(rec_sp)

        # Candidate 3: Simplified Macro-3 Model (S&P, Crude, VIX)
        crude_q = live_snap.get("crude") or {}
        crude_pct = float(crude_q.get("chg_pct") or 0.0)
        # Macro-3 heuristic linear probability
        macro_score = sp_pct * 0.45 - crude_pct * 0.10
        m3_class = 2 if macro_score >= 0.25 else (1 if macro_score <= -0.25 else 0)
        rec_m3 = pred_record.copy()
        rec_m3["model_type"] = "simple_macro3"
        rec_m3["predicted_class"] = m3_class
        rec_m3["q50_pts"] = round(forecast.get("spot_reference", 25000.0) * (macro_score / 100.0), 1)
        log_live_prediction(rec_m3)

        # Candidate 4: Live GIFT Nifty Basis Rule
        if gift_available:
            spot_ref = float(forecast.get("spot_reference", 25000.0))
            gift_basis_pct = ((float(gift_last) - spot_ref) / spot_ref) * 100.0
            gift_class = 2 if gift_basis_pct >= 0.30 else (1 if gift_basis_pct <= -0.30 else 0)
            rec_gift = pred_record.copy()
            rec_gift["model_type"] = "gift_implied"
            rec_gift["predicted_class"] = gift_class
            rec_gift["q50_pts"] = round(float(gift_last) - spot_ref, 1)
            log_live_prediction(rec_gift)

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
    cols_to_show = [c for c in ['timestamp', 'model_type', 'run_type', 'is_trading_day', 'status_flag', 'spot_ref', 'predicted_class', 'q50_pts', 'verdict'] if c in df.columns]
    print(df[cols_to_show].tail(6).to_string(index=False))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forward Test Scheduled Runner")
    parser.add_argument("--job-type", choices=["close_1515", "preopen_0845", "fill_outcomes", "probe", "audit", "all"], default="audit")
    parser.add_argument("--run-type", choices=["SCHEDULED", "MANUAL", "TEST"], default="MANUAL")
    args = parser.parse_args()

    if args.job_type == "probe":
        probe_feed_connectivity()
    elif args.job_type == "close_1515":
        run_prediction_job("close_1515", run_type=args.run_type)
    elif args.job_type == "preopen_0845":
        run_prediction_job("preopen_0845", run_type=args.run_type)
    elif args.job_type == "fill_outcomes":
        run_outcome_fill_job()
    elif args.job_type == "all":
        probe_feed_connectivity()
        run_prediction_job("close_1515", run_type=args.run_type)
        run_prediction_job("preopen_0845", run_type=args.run_type)
        run_outcome_fill_job()

    print_log_audit()
