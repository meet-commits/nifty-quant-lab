"""
dashboard/components/gap_forecast_card.py
Dual-Horizon Overnight Gap Forecast Component.
Replaces trading calls with neutral signals:
  - Model signal: gap-up bias
  - Model signal: gap-down bias
  - Model signal: no edge (flat/neutral)
Displays 3-class calibrated probabilities, quantile points [P10, P50, P90],
and Black-Scholes overnight option break-even hurdle.
"""

import html
import streamlit as st
import pandas as pd
from typing import Dict, Any

from gap_prediction_service import compute_overnight_gap_forecast, update_forward_test_outcomes

def render_gap_forecast_card(live_snap: Dict[str, Any]) -> None:
    """Renders the Kal Ka Overnight Gap Forecast card with neutral language and scoreboard expander."""
    fc_col1, fc_col2 = st.columns([2.5, 1.5])
    with fc_col1:
        selected_fc_mode = st.radio(
            "Forecast Horizon Model:",
            ["🕒 15:15 PM EOD Model (Strict Pre-15:30 IST Data)", "🌅 08:45 AM Pre-Open Nowcast (Overnight US + GIFT Basis)"],
            horizontal=True,
            label_visibility="collapsed"
        )
    with fc_col2:
        model_type_key = "eod_1515" if "15:15" in selected_fc_mode else "preopen_0845"
        manual_override = None
        if model_type_key == "preopen_0845":
            manual_override = st.number_input("Optional Manual GIFT Basis (pts):", value=0.0, step=5.0, help="Override futures basis spread if exchange feed is delayed")

    # Run inference
    gap_fc = compute_overnight_gap_forecast(live_snap, model_type=model_type_key, manual_gift_price=manual_override)

    if gap_fc.get("status") == "ERROR":
        st.warning(f"⚠️ Kal Ka Overnight Model: {gap_fc.get('error_message', 'Model offline')}")
    else:
        raw_p_up = gap_fc.get("prob_up_pct", "0%")
        raw_p_down = gap_fc.get("prob_down_pct", "0%")
        raw_p_flat = gap_fc.get("prob_flat_pct", "0%")

        try:
            p_up_num = float(str(raw_p_up).replace("%", "").strip() or 0.0)
            p_down_num = float(str(raw_p_down).replace("%", "").strip() or 0.0)
            p_flat_num = float(str(raw_p_flat).replace("%", "").strip() or 0.0)
        except (ValueError, TypeError):
            p_up_num, p_down_num, p_flat_num = 33.3, 33.3, 33.3

        p_up_str = f"{p_up_num:.1f}%"
        p_down_str = f"{p_down_num:.1f}%"
        p_flat_str = f"{p_flat_num:.1f}%"

        q10 = gap_fc.get("q10_pts", 0.0)
        q50 = gap_fc.get("q50_pts", 0.0)
        q90 = gap_fc.get("q90_pts", 0.0)
        
        raw_advice = gap_fc.get("action_advice", "")
        raw_signal = gap_fc.get("signal_badge", "")
        
        be_call = gap_fc.get("breakeven_call", {})
        be_put = gap_fc.get("breakeven_put", {})
        be_call_pts = be_call.get("breakeven_gap_pts", 35.0)
        be_put_pts = be_put.get("breakeven_gap_pts", 35.0)
        theta_drag = be_call.get("theta_decay_drag_pts", 12.0)
        iv_used = gap_fc.get("vix_reference", 15.0)

        # Map to neutral quantitative terminology (Task 5 requirement)
        if "CALL" in raw_advice or "GAP UP" in raw_signal:
            signal_badge = "Model signal: gap-up bias"
            signal_color = "#10B981"
        elif "PUT" in raw_advice or "GAP DOWN" in raw_signal:
            signal_badge = "Model signal: gap-down bias"
            signal_color = "#F43F5E"
        else:
            signal_badge = "Model signal: no edge (flat/neutral)"
            signal_color = "#F59E0B"

        horizon_label = '3:15 PM EOD HORIZON' if model_type_key == 'eod_1515' else '08:45 AM PRE-OPEN NOWCAST'

        st.html(f"""
        <div class="monolith-card-gold" style="padding:16px 20px; margin-bottom:18px; border:1px solid rgba(245,158,11,0.3);">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:12px;">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="font-size:20px;">🌙</span>
                    <span class="font-syncopate" style="font-size:12px; font-weight:700; color:#FFFFFF; letter-spacing:2px;">
                        OVERNIGHT GAP PROJECTION <span style="font-size:10px; color:#FCD34D;">({horizon_label})</span>
                    </span>
                </div>
                <div style="display:flex; gap:10px; align-items:center;">
                    <span style="font-family:'JetBrains Mono', monospace; font-size:11px; font-weight:800; color:{signal_color}; background:{signal_color}18; border:1px solid {signal_color}55; padding:4px 14px; border-radius:999px;">
                        ● {signal_badge}
                    </span>
                    <span style="font-family:'JetBrains Mono', monospace; font-size:10px; color:#94A3B8;">
                        IV: {iv_used:.1f}%
                    </span>
                </div>
            </div>

            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:12px; margin-bottom:14px;">
                <div style="background:rgba(0,0,0,0.5); padding:10px 14px; border-radius:10px; border:1px solid rgba(255,255,255,0.06);">
                    <div style="font-family:'JetBrains Mono', monospace; font-size:10px; color:#94A3B8; margin-bottom:4px;">3-CLASS CALIBRATED PROBABILITIES</div>
                    <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono', monospace; font-size:11px; font-weight:800;">
                        <span style="color:#F43F5E;">Down: {p_down_str}</span>
                        <span style="color:#F59E0B;">Flat: {p_flat_str}</span>
                        <span style="color:#10B981;">Up: {p_up_str}</span>
                    </div>
                    <div style="width:100%; height:6px; background:#1E293B; border-radius:999px; overflow:hidden; display:flex; margin-top:6px;">
                        <div style="width:{p_down_num}%; background:#F43F5E;"></div>
                        <div style="width:{p_flat_num}%; background:#F59E0B;"></div>
                        <div style="width:{p_up_num}%; background:#10B981;"></div>
                    </div>
                </div>

                <div style="background:rgba(0,0,0,0.5); padding:10px 14px; border-radius:10px; border:1px solid rgba(255,255,255,0.06);">
                    <div style="font-family:'JetBrains Mono', monospace; font-size:10px; color:#94A3B8; margin-bottom:4px;">QUANTILE REGRESSION GAP RANGE</div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:14px; font-weight:800; color:#FFFFFF;">
                        {q50:+.1f} pts <span style="font-size:10.5px; color:#FCD34D;">[P10: {q10:+.1f} | P90: {q90:+.1f}]</span>
                    </div>
                    <div style="font-size:9.5px; color:#64748B; margin-top:3px;">Median prediction with 80% confidence interval</div>
                </div>

                <div style="background:rgba(0,0,0,0.5); padding:10px 14px; border-radius:10px; border:1px solid rgba(255,255,255,0.06);">
                    <div style="font-family:'JetBrains Mono', monospace; font-size:10px; color:#94A3B8; margin-bottom:4px;">BLACK-SCHOLES THETA BREAK-EVEN</div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:11.5px; font-weight:700; color:#CBD5E1;">
                        Call: <b style="color:#10B981;">+{be_call_pts:.1f} pts</b> | Put: <b style="color:#F43F5E;">-{be_put_pts:.1f} pts</b>
                    </div>
                    <div style="font-size:9.5px; color:#64748B; margin-top:3px;">Theta cost drag: ~{theta_drag:.1f} pts</div>
                </div>
            </div>
        </div>
        """)

    # Scoreboards Expander
    with st.expander("📋 View Verifiable Prediction Scoreboards (Backtest vs Real Forward-Test Log)", expanded=False):
        sb_tab1, sb_tab2 = st.tabs(["📋 Walk-Forward Out-of-Sample Backtest", "⚡ Real Live Forward-Test Log"])
        with sb_tab1:
            st.caption("Historical out-of-sample predictions generated on unseen test sets across ~10 years:")
            if "scoreboard_df" in gap_fc and gap_fc["scoreboard_df"] is not None:
                st.dataframe(gap_fc["scoreboard_df"], use_container_width=True, hide_index=True)
        with sb_tab2:
            st.caption("Live forward-test log saved to `data/predictions_log.parquet`. Contains all real predictions logged in production:")
            try:
                fwd_df = update_forward_test_outcomes()
                if fwd_df is not None and not fwd_df.empty:
                    st.dataframe(fwd_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No live forward-test logs recorded yet. Real predictions will appear here as live sessions complete.")
            except Exception as e:
                st.info(f"Forward-test log awaiting completed sessions ({e}).")
