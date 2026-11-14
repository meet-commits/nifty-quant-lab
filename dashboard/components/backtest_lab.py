"""
dashboard/components/backtest_lab.py
Expanding Walk-Forward Validation & Black-Scholes Friction Lab.
Stores backtest results in st.session_state so they survive all reruns and fragment ticks.
"""

import streamlit as st
import plotly.express as px

from feature_engine import build_feature_matrix
from model_engine import train_and_evaluate_walk_forward
from backtest_engine import run_realistic_backtest

def render_backtest_lab() -> None:
    """Renders Walk-Forward Backtest Lab with persistent session state."""
    st.html("<div class='section-header'><span>05 / WALK-FORWARD VALIDATION & STATUTORY POST-FRICTION LAB</span></div>")
    st.html("""
    <div class="monolith-card" style="margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-family:'Syncopate', sans-serif; font-size:12px; font-weight:700; color:#FFFFFF;">
                    EXPANDING WALK-FORWARD ACCURACY & BLACK-SCHOLES FRICTION AUDIT
                </div>
                <div style="font-size:12px; color:#94A3B8; margin-top:2px;">
                    Strict expanding-window walk-forward evaluation on 2,000+ out-of-sample historical sessions (~10-year DuckDB archive). Options pricing evaluated via continuous Black-Scholes model (ATM strike, India VIX as IV, calendar-day theta decay, and statutory STT/GST/charges loaded from config/charges.yaml).
                </div>
            </div>
        </div>
    </div>
    """)

    # Check if backtest state exists
    if "wf_backtest_data" not in st.session_state:
        st.session_state["wf_backtest_data"] = None

    c_btn, c_clear = st.columns([3, 1])
    with c_btn:
        run_clicked = st.button("🚀 Run Expanding Walk-Forward Validation On Historical Archive (~10 Years)", type="primary", use_container_width=True)
    with c_clear:
        if st.session_state["wf_backtest_data"] is not None:
            if st.button("🗑️ Clear Stored Results", use_container_width=True):
                st.session_state["wf_backtest_data"] = None
                st.rerun()

    if run_clicked:
        with st.spinner("Executing expanding walk-forward folds and calculating Black-Scholes options equity curve..."):
            try:
                df_features = build_feature_matrix()
                res_df, model_metrics = train_and_evaluate_walk_forward(df=df_features, initial_train_window=600, step_size=50)
                trades_df, b_summary = run_realistic_backtest(predictions_df=res_df)
                st.session_state["wf_backtest_data"] = {
                    "metrics": model_metrics,
                    "summary": b_summary,
                    "trades": trades_df
                }
                st.success("✅ Walk-forward validation completed and cached in session.")
            except Exception as e:
                st.error(f"Backtest execution error: {e}")

    # Render persisted results from session state
    saved_data = st.session_state.get("wf_backtest_data")
    if saved_data:
        m = saved_data["metrics"]
        b = saved_data["summary"]
        trades_df = saved_data["trades"]

        c_m1, c_m2, c_m3, c_m4, c_m5 = st.columns(5)
        c_m1.metric(
            "3-Class Accuracy",
            f"{m['model_accuracy']}%",
            delta=f"{m['model_accuracy'] - m['baseline_majority_acc']:+.1f}% vs Prior"
        )
        c_m2.metric(
            "Brier Calibration",
            f"{m['model_brier_score']}",
            delta=f"{m['model_brier_score'] - m['baseline_brier_prior']:.4f} vs Prior",
            delta_color="inverse"
        )
        c_m3.metric("Black-Scholes Net Return", f"{b['net_return_pct']}%")
        c_m4.metric("Post-Friction Win Rate", f"{b['win_rate_pct']}%")
        c_m5.metric("Statutory Friction Paid", f"₹{b['total_friction_paid_inr']:,.0f}")

        fig_eq = px.line(trades_df, x="date", y="capital_after_trade", title="Net Equity Curve (Black-Scholes Greeks + Statutory Indian Friction)")
        fig_eq.update_layout(template="plotly_dark", height=280, paper_bgcolor='#000000', plot_bgcolor='#000000')
        st.plotly_chart(fig_eq, use_container_width=True)

