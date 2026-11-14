"""
dashboard/components/intraday_section.py
Intraday Convergence & Live 5m Candlestick Chart.
Renames Tug-of-War to "Intraday Strength Score (heuristic)" with exact formula tooltip.
"""

import html
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any

from live_market_service import get_live_intraday_candles

def render_intraday_section(selected_asset: str, current: Dict[str, Any], live_snap: Dict[str, Any]) -> None:
    """Renders the Intraday Strength Score card and live 5m candlestick chart."""
    st.html("<div class='section-header'><span>01 / INTRADAY CONVERGENCE & LIVE CHART</span></div>")

    tow_label = "BULLS LEADING" if current['bull_pct'] >= 60 else ("BEARS LEADING" if current['bear_pct'] >= 60 else "NEUTRAL MOMENTUM")
    tow_color = "#10B981" if current['bull_pct'] >= 50 else "#F43F5E"

    vix_q_data = live_snap.get("vix", {})
    crude_q_data = live_snap.get("crude", {})
    usd_q_data = live_snap.get("usdinr", {})
    sp_q_data = live_snap.get("sp500", {})

    vix_badge = "LIVE" if vix_q_data.get("is_live", False) else ("STALE" if vix_q_data.get("is_stale", False) else "POST-MKT")
    crude_badge = "LIVE" if crude_q_data.get("is_live", False) else ("STALE" if crude_q_data.get("is_stale", False) else "GLOBAL FUT")
    usd_badge = "LIVE" if usd_q_data.get("is_live", False) else ("STALE" if usd_q_data.get("is_stale", False) else "FOREX EOD")
    sp_badge = "LIVE" if sp_q_data.get("is_live", False) else ("STALE" if sp_q_data.get("is_stale", False) else "US GLOBEX")

    vix_close = vix_q_data.get("value") or vix_q_data.get("last")
    vix_pct = vix_q_data.get("chg_pct")
    crude_close = crude_q_data.get("value") or crude_q_data.get("last")
    crude_pct = crude_q_data.get("chg_pct")
    usd_close = usd_q_data.get("value") or usd_q_data.get("last")
    usd_pct = usd_q_data.get("chg_pct")
    sp_close = sp_q_data.get("value") or sp_q_data.get("last")
    sp_pct = sp_q_data.get("chg_pct")

    vix_disp = f"{vix_close:.2f} ({vix_pct:+.2f}%)" if (vix_close is not None and vix_pct is not None) else "N/A"
    crude_disp = f"${crude_close:.2f} ({crude_pct:+.2f}%)" if (crude_close is not None and crude_pct is not None) else "N/A"
    usd_disp = f"₹{usd_close:.2f} ({usd_pct:+.2f}%)" if (usd_close is not None and usd_pct is not None) else "N/A"
    sp_disp = f"{sp_close:,.1f} ({sp_pct:+.2f}%)" if (sp_close is not None and sp_pct is not None) else "N/A"

    col_tow, col_chart = st.columns([1, 2])

    with col_tow:
        st.html(f"""
        <div class="monolith-card" style="height:100%;">
            <div style="display:flex; justify-content:space-between; align-items:center; padding-bottom:12px; border-bottom:1px solid rgba(255,255,255,0.06);">
                <span class="font-syncopate" title="Formula: 40% Range Position + 35% % Change + 25% Constituent Breadth" style="font-size:11px; font-weight:700; color:#FFFFFF; cursor:help;">
                    ⚖️ Strength Score (heuristic)
                </span>
                <span style="font-family:'JetBrains Mono', monospace; font-size:11px; font-weight:800; color:{tow_color};">{tow_label}</span>
            </div>
            
            <div style="margin:16px 0 12px 0;">
                <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono', monospace; font-size:12px; font-weight:800; margin-bottom:6px;">
                    <span style="color:#10B981;">{current['bull_pct']}% BULL FACTOR</span>
                    <span style="color:#F43F5E;">{current['bear_pct']}% BEAR FACTOR</span>
                </div>
                <div style="width:100%; background:#000000; height:18px; border-radius:999px; overflow:hidden; display:flex; border:1px solid rgba(255,255,255,0.1);">
                    <div style="width:{current['bull_pct']}%; background:linear-gradient(90deg, #10B981, #00F0FF);"></div>
                    <div style="width:{current['bear_pct']}%; background:#F43F5E;"></div>
                </div>
                <div style="font-size:9.5px; color:#64748B; margin-top:4px; text-align:right;">
                    ℹ️ Mathematical momentum metric; not a directional trade call.
                </div>
            </div>

            <div style="font-family:'JetBrains Mono', monospace; font-size:11px; color:#CBD5E1; margin-top:16px;">
                <div style="background:rgba(0,0,0,0.4); padding:8px 12px; border-radius:10px; border:1px solid rgba(255,255,255,0.05); margin-bottom:6px; display:flex; justify-content:space-between;">
                    <span>India VIX <b style="color:#94A3B8; font-size:9.5px;">[{vix_badge}]</b>:</span>
                    <b style="color:#10B981;">{vix_disp}</b>
                </div>
                <div style="background:rgba(0,0,0,0.4); padding:8px 12px; border-radius:10px; border:1px solid rgba(255,255,255,0.05); margin-bottom:6px; display:flex; justify-content:space-between;">
                    <span>Crude WTI Oil <b style="color:#94A3B8; font-size:9.5px;">[{crude_badge}]</b>:</span>
                    <b style="color:#10B981;">{crude_disp}</b>
                </div>
                <div style="background:rgba(0,0,0,0.4); padding:8px 12px; border-radius:10px; border:1px solid rgba(255,255,255,0.05); margin-bottom:6px; display:flex; justify-content:space-between;">
                    <span>USD/INR Spot <b style="color:#94A3B8; font-size:9.5px;">[{usd_badge}]</b>:</span>
                    <b style="color:#10B981;">{usd_disp}</b>
                </div>
                <div style="background:rgba(0,0,0,0.4); padding:8px 12px; border-radius:10px; border:1px solid rgba(255,255,255,0.05); display:flex; justify-content:space-between;">
                    <span>US S&P 500 Fut <b style="color:#94A3B8; font-size:9.5px;">[{sp_badge}]</b>:</span>
                    <b style="color:#10B981;">{sp_disp}</b>
                </div>
            </div>
        </div>
        """)

    with col_chart:
        is_daily_bar_fallback = False
        n_close = (live_snap.get("nifty") or {}).get("value")
        g_close = (live_snap.get("giftnifty") or {}).get("value")
        g_is_real = (live_snap.get("giftnifty") or {}).get("is_real", False)

        if selected_asset == 'NIFTY 50':
            chart_sym = '^NSEI'
            chart_df = get_live_intraday_candles(chart_sym, '5m')
            is_daily_bar_fallback = chart_df.attrs.get("is_daily_fallback", False)
            chart_name = "NIFTY 50 (5m Spot)" if not is_daily_bar_fallback else "NIFTY 50 (Daily Bar Archive)"
        elif selected_asset == 'BANK NIFTY':
            chart_sym = '^NSEBANK'
            chart_df = get_live_intraday_candles(chart_sym, '5m')
            is_daily_bar_fallback = chart_df.attrs.get("is_daily_fallback", False)
            chart_name = "BANK NIFTY (5m Spot)" if not is_daily_bar_fallback else "BANK NIFTY (Daily Bar Archive)"
        elif selected_asset == 'GIFT NIFTY':
            chart_sym = '^NSEI'
            chart_df = get_live_intraday_candles(chart_sym, '5m')
            is_daily_bar_fallback = chart_df.attrs.get("is_daily_fallback", False)
            basis = round((g_close or 0.0) - (n_close or 0.0), 2) if (g_close and n_close) else 0.0
            if not chart_df.empty:
                chart_df = chart_df.copy()
                for col in ['Open', 'High', 'Low', 'Close']:
                    chart_df[col] = (chart_df[col] + basis).round(2)
            chart_name = f"GIFT NIFTY ({'Official Derivative Track' if g_is_real else 'Approximation: Spot + Basis'})"
        else:
            chart_sym = '^BSESN'
            chart_df = get_live_intraday_candles(chart_sym, '5m')
            is_daily_bar_fallback = chart_df.attrs.get("is_daily_fallback", False)
            chart_name = "SENSEX (5m BSE)" if not is_daily_bar_fallback else "SENSEX (Daily Bar Archive)"

        if is_daily_bar_fallback:
            st.caption("ℹ️ *Note: Intraday 5m feed closed/unavailable. Showing completed daily session bars from verified DuckDB archive.*")
        elif selected_asset == 'GIFT NIFTY':
            st.caption("ℹ️ *Note: GIFT Nifty candle track is calibrated from Spot candles adjusted for live futures basis spread.*")

        if not chart_df.empty and 'Close' in chart_df.columns:
            fig_candle = go.Figure(data=[go.Candlestick(
                x=chart_df.index,
                open=chart_df['Open'],
                high=chart_df['High'],
                low=chart_df['Low'],
                close=chart_df['Close'],
                name=chart_name,
                increasing_line_color='#10B981',
                decreasing_line_color='#F43F5E'
            )])

            # Floor Pivots overlay
            sl_val = current.get('sl')
            piv_val = current.get('pivot')
            t1_val = current.get('t1')

            if piv_val:
                fig_candle.add_hline(y=piv_val, line_dash="dash", line_color="#F59E0B", annotation_text="Pivot (P)", annotation_position="top left")
            if sl_val:
                fig_candle.add_hline(y=sl_val, line_dash="dot", line_color="#F43F5E", annotation_text="Ref S1", annotation_position="bottom left")
            if t1_val:
                fig_candle.add_hline(y=t1_val, line_dash="dash", line_color="#10B981", annotation_text="Ref R1", annotation_position="top left")

            fig_candle.update_layout(
                template="plotly_dark",
                height=340,
                margin=dict(l=10, r=10, t=25, b=10),
                xaxis_rangeslider_visible=False,
                paper_bgcolor='#000000',
                plot_bgcolor='#000000',
                title=dict(text=chart_name, font=dict(family='JetBrains Mono', size=12, color='#94A3B8'))
            )
            st.plotly_chart(fig_candle, use_container_width=True)
        else:
            st.warning("Candle data offline for this asset. Awaiting market feed ticks.")

