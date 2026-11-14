"""
scripts/forward_test_runner.py
Scheduled Forward-Test Automation Engine.
Runs independently of Streamlit Cloud in GitHub Actions:
- 15:15 IST (09:45 UTC): Generates Close Model (eod_1515) prediction record.
- 08:45 IST (03:15 UTC): Generates Pre-Open Nowcast (preopen_0845) prediction record.
- 09:30 IST (04:00 UTC): Fills in actual next-day open gap outcomes for pending predictions.
Saves records idempotently to data/predictions_log.parquet and commits to data-archive branch.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from src.live_market_service import fetch_live_market_snapshot
from src.gap_prediction_service import (
    compute_overnight_gap_forecast,
    update_forward_test_outcomes,
    PREDICTIONS_LOG_PATH
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FORWARD_TEST_RUNNER")
IST = ZoneInfo("Asia/Kolkata")


def run_prediction_job(model_type: str = "eod_1515"):
    now_ist = datetime.now(IST)
    logger.info(f"🚀 Running Scheduled Prediction Job: {model_type.upper()} at {now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}")

    # Fetch live snapshot
    live_snap = fetch_live_market_snapshot()
    if not live_snap:
        logger.error("❌ Failed to fetch live snapshot from data feeds.")
        return False

    # Generate prediction and append to predictions_log.parquet
    forecast = compute_overnight_gap_forecast(live_snap, model_type=model_type)
    logger.info(f"✅ Prediction generated: {forecast.get('signal_badge')} | Prob UP: {forecast.get('prob_up_pct')}% | Prob DOWN: {forecast.get('prob_down_pct')}%")
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
    print("\n" + "=" * 80)
    print("FORWARD-TEST LOG AUDIT EVIDENCE")
    print("=" * 80)
    print(f"Total Logged Predictions : {len(df)}")
    print(f"Min Timestamp            : {df['timestamp'].min() if 'timestamp' in df.columns else 'N/A'}")
    print(f"Max Timestamp            : {df['timestamp'].max() if 'timestamp' in df.columns else 'N/A'}")
    print(f"Models Logged            : {df['model_type'].unique().tolist() if 'model_type' in df.columns else 'N/A'}")
    print(f"Verdicts Distribution    :\n{df['verdict'].value_counts() if 'verdict' in df.columns else 'N/A'}")
    print("\nRecent 5 Forward-Test Records:")
    cols_to_show = [c for c in ['timestamp', 'model_type', 'spot_ref', 'predicted_class', 'prob_up', 'prob_down', 'q50_pts', 'actual_gap_pts', 'verdict'] if c in df.columns]
    print(df[cols_to_show].tail(5).to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forward Test Scheduled Runner")
    parser.add_argument("--job-type", choices=["close_1515", "preopen_0845", "fill_outcomes", "audit", "all"], default="audit")
    args = parser.parse_args()

    if args.job_type == "close_1515":
        run_prediction_job("eod_1515")
    elif args.job_type == "preopen_0845":
        run_prediction_job("preopen_0845")
    elif args.job_type == "fill_outcomes":
        run_outcome_fill_job()
    elif args.job_type == "all":
        run_prediction_job("eod_1515")
        run_outcome_fill_job()

    print_log_audit()
