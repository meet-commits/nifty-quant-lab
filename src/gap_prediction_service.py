"""
src/gap_prediction_service.py
Dedicated 3:15 PM EOD & 08:45 Pre-Open Overnight Gap Predictor, Quantile Gap Forecaster,
Black-Scholes Break-Even Engine, and Verifiable Forward-Test Scoreboard.
"""

import os
import duckdb
import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Any, Optional

try:
    import streamlit as st
    cache_resource_decorator = st.cache_resource(ttl=86400)
except Exception:
    def cache_resource_decorator(func):
        return func

from feature_engine import (
    build_feature_matrix,
    extract_live_feature_vector
)
from model_engine import (
    train_and_evaluate_walk_forward,
    fit_final_production_models
)
from backtest_engine import compute_option_breakeven

IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
PREDICTIONS_LOG_PATH = os.path.join(DATA_DIR, "predictions_log.parquet")
DB_PATH = os.path.join(DATA_DIR, "research.duckdb")
MARKET_DATA_PATH = os.path.join(DATA_DIR, "market_data.parquet")
SUITE_CACHE_PATH = os.path.join(DATA_DIR, "production_suite.joblib")


@cache_resource_decorator
def get_cached_production_suite() -> Dict[str, Any]:
    """
    Loads pre-trained production model suite from disk if valid;
    otherwise trains expanding-window walk-forward evaluation and saves to disk.
    """
    if os.path.exists(SUITE_CACHE_PATH):
        try:
            mtime_cache = os.path.getmtime(SUITE_CACHE_PATH)
            mtime_data = os.path.getmtime(MARKET_DATA_PATH) if os.path.exists(MARKET_DATA_PATH) else 0.0
            if mtime_cache >= mtime_data:
                import joblib
                return joblib.load(SUITE_CACHE_PATH)
        except Exception as e:
            print(f"⚠️ Cache read error, retraining: {e}")

    print("🧠 Initializing Production Gap Prediction Suite (Expanding Walk-Forward & Full Refit)...")
    df = build_feature_matrix()

    # 1. Rigorous walk-forward validation across history
    res_df, metrics = train_and_evaluate_walk_forward(df=df, initial_train_window=600, step_size=50)

    # 2. Refit final production models on 100% of data (Zero wasted training data)
    prod_eod = fit_final_production_models(df=df, model_type="eod_1515")
    prod_preopen = fit_final_production_models(df=df, model_type="preopen_0845")

    # 3. Build Backtest Scoreboard (Most recent 15 out-of-sample sessions)
    backtest_rows = []
    for i in range(max(0, len(res_df) - 15), len(res_df)):
        r = res_df.iloc[i]
        p_up = r['predicted_prob_up']
        p_down = r['predicted_prob_down']
        p_flat = r['predicted_prob_flat']
        act_class = int(r['actual_target_3class'])
        act_gap = r['actual_gap_pct']
        act_pts = r['actual_gap_pts']

        pred_class = int(r['predicted_class'])
        hit = (pred_class == act_class)

        class_labels = {2: "🟢 GAP UP (>+0.25%)", 1: "🔴 GAP DOWN (<-0.25%)", 0: "🟡 FLAT / CHOP"}
        backtest_rows.append({
            "Date": pd.to_datetime(r['date']).strftime("%d %b %Y"),
            "Predicted_Regime": class_labels.get(pred_class, "UNKNOWN"),
            "P(Up)": f"{p_up*100:.1f}%",
            "P(Flat)": f"{p_flat*100:.1f}%",
            "P(Down)": f"{p_down*100:.1f}%",
            "Expected_Range (P10-P90)": f"{r['pred_q10_pts']:+.0f} to {r['pred_q90_pts']:+.0f} pts",
            "Actual_Gap": f"{'+' if act_gap>=0 else ''}{act_gap:.2f}% ({'+' if act_pts>=0 else ''}{act_pts:.1f} pts)",
            "Verdict": "🎯 HIT" if hit else "⚠️ MISS"
        })

    backtest_scoreboard_df = pd.DataFrame(backtest_rows).iloc[::-1]

    suite = {
        "metrics": metrics,
        "walkforward_df": res_df,
        "backtest_scoreboard_df": backtest_scoreboard_df,
        "prod_eod": prod_eod,
        "prod_preopen": prod_preopen
    }

    try:
        import joblib
        joblib.dump(suite, SUITE_CACHE_PATH)
    except Exception as e:
        print(f"⚠️ Could not persist suite to disk: {e}")

    return suite


def update_forward_test_outcomes() -> pd.DataFrame:
    """
    Checks historical DuckDB completed sessions to fill in actual next-day outcomes
    for logged live predictions that are still PENDING.
    """
    if not os.path.exists(PREDICTIONS_LOG_PATH):
        return pd.DataFrame()

    try:
        log_df = pd.read_parquet(PREDICTIONS_LOG_PATH)
        if log_df.empty or "verdict" not in log_df.columns:
            return log_df

        # If any rows are PENDING, try to resolve via DuckDB daily_prices
        pending_mask = log_df['verdict'].str.contains("PENDING", na=False)
        if pending_mask.any() and os.path.exists(DB_PATH):
            con = duckdb.connect(DB_PATH, read_only=True)
            daily_nifty = con.execute("SELECT date, open, close FROM daily_prices WHERE symbol='NIFTY_SPOT' ORDER BY date ASC;").df()
            con.close()
            daily_nifty['date_str'] = pd.to_datetime(daily_nifty['date']).dt.strftime('%Y-%m-%d')

            updated = False
            for idx in log_df[pending_mask].index:
                pred_date = str(log_df.loc[idx, 'date'])
                # Find the next trading day after pred_date
                future_days = daily_nifty[daily_nifty['date_str'] > pred_date]
                if not future_days.empty:
                    next_day = future_days.iloc[0]
                    next_open = float(next_day['open'])
                    spot_ref = float(log_df.loc[idx, 'spot_ref'])
                    gap_pts = next_open - spot_ref
                    gap_pct = (gap_pts / spot_ref) * 100.0

                    if gap_pct >= 0.25:
                        actual_regime = 2
                    elif gap_pct <= -0.25:
                        actual_regime = 1
                    else:
                        actual_regime = 0

                    pred_class = int(log_df.loc[idx, 'predicted_class'])
                    hit = (pred_class == actual_regime)

                    log_df.loc[idx, 'actual_gap_pct'] = round(gap_pct, 2)
                    log_df.loc[idx, 'actual_gap_pts'] = round(gap_pts, 1)
                    log_df.loc[idx, 'verdict'] = "🎯 HIT" if hit else "⚠️ MISS"
                    updated = True

            if updated:
                log_df.to_parquet(PREDICTIONS_LOG_PATH, index=False)

        return log_df
    except Exception as e:
        print(f"⚠️ Error resolving forward test log: {e}")
        return pd.DataFrame()


def log_live_prediction(prediction_record: Dict[str, Any]) -> None:
    """Appends live prediction record to data/predictions_log.parquet (idempotent per date/model)."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        if os.path.exists(PREDICTIONS_LOG_PATH):
            df_log = pd.read_parquet(PREDICTIONS_LOG_PATH)
        else:
            df_log = pd.DataFrame()

        new_row = pd.DataFrame([prediction_record])

        # Avoid duplicate log entries for the same date and model_type
        if not df_log.empty and 'date' in df_log.columns and 'model_type' in df_log.columns:
            mask = (df_log['date'] == prediction_record['date']) & (df_log['model_type'] == prediction_record['model_type'])
            df_log = df_log[~mask]
            df_log = pd.concat([df_log, new_row], ignore_index=True)
        else:
            df_log = new_row

        df_log.to_parquet(PREDICTIONS_LOG_PATH, index=False)
    except Exception as e:
        print(f"⚠️ Error saving live prediction log: {e}")


def compute_overnight_gap_forecast(
    live_snap: Dict[str, Any],
    model_type: str = "eod_1515",
    manual_gift_price: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes overnight gap probability (3-class), quantile point range (P10, P50, P90),
    and Black-Scholes theta break-even hurdle from today's live snapshot.
    """
    suite = get_cached_production_suite()
    metrics = suite["metrics"]
    backtest_scoreboard_df = suite["backtest_scoreboard_df"]

    prod_model_info = suite["prod_eod"] if model_type == "eod_1515" else suite["prod_preopen"]
    clf = prod_model_info["classifier"]
    scaler = prod_model_info["scaler"]
    qr_10 = prod_model_info["qr_10"]
    qr_50 = prod_model_info["qr_50"]
    qr_90 = prod_model_info["qr_90"]

    now_ist = datetime.now(IST)
    today_str = now_ist.strftime("%Y-%m-%d")

    # 1. Extract Live Feature Vector via identical feature path
    try:
        # Override GIFT price if manually supplied by user
        if manual_gift_price and manual_gift_price > 1000.0:
            if "giftnifty" not in live_snap:
                live_snap["giftnifty"] = {}
            live_snap["giftnifty"]["last"] = float(manual_gift_price)

        live_features_df, live_symbols = extract_live_feature_vector(live_snap, model_type=model_type)
        spot_ref = float(live_symbols["NIFTY_SPOT"]["close"])
        vix_ref = float(live_symbols["INDIA_VIX"]["close"])
    except Exception as e:
        # Honest Error State: Never fallback silently to fake 50%
        return {
            "status": "ERROR",
            "error_message": f"MODEL ERROR: {str(e)}",
            "signal_badge": "⚠️ MODEL ERROR",
            "signal_color": "#F43F5E",
            "action_advice": "DATA UNAVAILABLE • DO NOT TRADE",
            "prob_up_pct": "N/A",
            "prob_flat_pct": "N/A",
            "prob_down_pct": "N/A",
            "expected_range_str": "N/A",
            "spot_reference": 0.0,
            "catalysts": [f"Feature Pipeline Error: {e}"],
            "metrics": metrics,
            "scoreboard_df": backtest_scoreboard_df,
            "forward_test_scoreboard_df": update_forward_test_outcomes(),
            "breakeven": {
                "breakeven_gap_pts": "N/A",
                "entry_premium": "N/A",
                "overnight_theta_loss_pts": "N/A"
            }
        }

    # 2. Model Inference
    X_live_scaled = scaler.transform(live_features_df.values)
    probs = clf.predict_proba(X_live_scaled)[0]  # [P(Flat), P(Down), P(Up)]
    p_flat = float(probs[0])
    p_down = float(probs[1])
    p_up = float(probs[2])

    pred_class = int(np.argmax(probs))

    # Quantile Point Predictions
    q10_pts = round(float(qr_10.predict(X_live_scaled)[0]), 1)
    q50_pts = round(float(qr_50.predict(X_live_scaled)[0]), 1)
    q90_pts = round(float(qr_90.predict(X_live_scaled)[0]), 1)

    prob_up_pct = round(p_up * 100.0, 1)
    prob_flat_pct = round(p_flat * 100.0, 1)
    prob_down_pct = round(p_down * 100.0, 1)

    # 3. Black-Scholes Break-Even Analysis
    cal_days = 3 if now_ist.weekday() == 4 else 1  # 3 days over weekend
    days_to_thursday = (3 - now_ist.weekday()) % 7
    dte = max(days_to_thursday, 1.0)

    be_call = compute_option_breakeven(spot=spot_ref, vix=vix_ref, days_to_expiry=dte, calendar_days_held=cal_days, option_type="call")
    be_put = compute_option_breakeven(spot=spot_ref, vix=vix_ref, days_to_expiry=dte, calendar_days_held=cal_days, option_type="put")

    # 4. Action Signal & Regimes
    expected_range_str = f"{q10_pts:+.0f} to {q90_pts:+.0f} pts [Median: {q50_pts:+.0f}]"

    if p_up >= 0.45 and q50_pts > be_call['breakeven_gap_pts']:
        signal_badge = "🟢 GAP UP (COVERS THETA HURDLE)"
        signal_color = "#10B981"
        action_advice = f"OVERNIGHT CALL FAVORED • Expected move (+{q50_pts:.0f} pts) beats break-even ({be_call['breakeven_gap_pts']:.0f} pts)"
    elif p_down >= 0.35 and abs(q50_pts) > be_put['breakeven_gap_pts']:
        signal_badge = "🔴 GAP DOWN (COVERS THETA HURDLE)"
        signal_color = "#F43F5E"
        action_advice = f"OVERNIGHT PUT FAVORED • Expected drop ({q50_pts:.0f} pts) beats break-even ({be_put['breakeven_gap_pts']:.0f} pts)"
    else:
        signal_badge = "🟡 FLAT / CHOP (THETA DANGER)"
        signal_color = "#F59E0B"
        action_advice = f"AVOID OVERNIGHT CARRY • Expected move fails to overcome {max(be_call['breakeven_gap_pts'], be_put['breakeven_gap_pts']):.0f} pt theta hurdle"

    # 5. Explanatory Catalysts
    catalysts = []
    range_pos = float(live_features_df.get('nifty_range_pos', pd.Series([0.5])).iloc[0])
    if range_pos >= 0.65:
        catalysts.append(f"Institutional Close Strength: Nifty closed near day's high ({range_pos*100:.0f}% range position).")
    elif range_pos <= 0.35:
        catalysts.append(f"Heavy EOD Supply: Nifty closed near day's low ({range_pos*100:.0f}% range position).")
    else:
        catalysts.append(f"Neutral Equilibrium: Nifty closed at mid-range ({range_pos*100:.0f}% range position).")

    vix_above = float(live_features_df.get('vix_above_ma5', pd.Series([0.0])).iloc[0])
    if vix_above > 0.5:
        catalysts.append("India VIX Elevated: Volatility is trending above its 5-day moving average (Option sellers charging higher premiums).")
    else:
        catalysts.append("India VIX Subdued: Volatility below 5-day average indicates low systemic hedging stress.")

    if model_type == "preopen_0845":
        g_basis = float(live_features_df.get('gift_basis_pct', pd.Series([0.0])).iloc[0])
        catalysts.append(f"Pre-Open GIFT Basis: {g_basis:+.2f}% spread relative to previous Spot close.")

    # 6. Log live prediction to forward-test archive
    pred_record = {
        "timestamp": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "date": today_str,
        "model_type": model_type,
        "spot_ref": spot_ref,
        "vix_ref": vix_ref,
        "predicted_class": pred_class,
        "prob_up": round(p_up, 4),
        "prob_flat": round(p_flat, 4),
        "prob_down": round(p_down, 4),
        "q10_pts": q10_pts,
        "q50_pts": q50_pts,
        "q90_pts": q90_pts,
        "breakeven_call_pts": be_call['breakeven_gap_pts'],
        "breakeven_put_pts": be_put['breakeven_gap_pts'],
        "actual_gap_pct": None,
        "actual_gap_pts": None,
        "verdict": "⏳ PENDING NEXT SESSION OPEN"
    }
    log_live_prediction(pred_record)

    # 7. Update and retrieve forward-test scoreboard
    forward_test_df = update_forward_test_outcomes()

    return {
        "status": "SUCCESS",
        "signal_badge": signal_badge,
        "signal_color": signal_color,
        "action_advice": action_advice,
        "prob_up_pct": prob_up_pct,
        "prob_flat_pct": prob_flat_pct,
        "prob_down_pct": prob_down_pct,
        "expected_range_str": expected_range_str,
        "q10_pts": q10_pts,
        "q50_pts": q50_pts,
        "q90_pts": q90_pts,
        "spot_reference": spot_ref,
        "vix_reference": vix_ref,
        "catalysts": catalysts,
        "metrics": metrics,
        "breakeven_call": be_call,
        "breakeven_put": be_put,
        "scoreboard_df": backtest_scoreboard_df,
        "forward_test_scoreboard_df": forward_test_df
    }


if __name__ == "__main__":
    suite = get_cached_production_suite()
    print("Metrics Preview:", suite["metrics"])
