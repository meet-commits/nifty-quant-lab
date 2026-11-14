"""
dashboard/components/chartink_suite.py
Chartink Scanner and Event Audit components.
Uses st.column_config for native sorting/formatting and escapes external strings.
"""

import html
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any

from event_calendar import get_high_impact_events
from news_service import fetch_live_market_news
from chartink_engine import (
    fetch_stock_real_ohlc,
    CHARTINK_CATEGORIES,
    get_market_breadth_stats,
    get_sector_heatmap
)
from bhavcopy_service import check_bhavcopy_backfill_health

def render_events_and_news() -> None:
    """Renders Section 04: Dynamic Event Countdown & News Dossier."""
    st.html("<div class='section-header'><span>04 / CHARTINK SCANNERS & EVENT AUDIT</span></div>")

    col_event, col_news = st.columns([1, 1.2])

    with col_event:
        events = get_high_impact_events()
        event_html_items = ""
        for ev in events[:4]:
            ev_name = html.escape(str(ev.get('event', '')))
            ev_date = html.escape(str(ev.get('date', '')))
            ev_impact = html.escape(str(ev.get('impact', '')))
            ev_color = ev.get('color', '#10B981')
            ev_countdown = html.escape(str(ev.get('countdown', '')))

            event_html_items += f"""
            <div style="display:flex; justify-content:space-between; margin-bottom:8px; padding-bottom:6px; border-bottom:1px solid rgba(255,255,255,0.05);">
                <div>
                    <span style="color:#F8FAFC; font-weight:600;">{ev_name}</span>
                    <div style="color:#64748B; font-size:10px;">{ev_date} • {ev_impact}</div>
                </div>
                <b style="color:{ev_color}; font-size:11px; align-self:center;">{ev_countdown}</b>
            </div>
            """
        st.html(f"""
        <div class="monolith-card" style="padding:16px; height:100%;">
            <div style="font-family:'Syncopate', sans-serif; font-size:11px; font-weight:700; color:#FFFFFF; margin-bottom:12px;">
                📅 DYNAMIC EVENT COUNTDOWN
            </div>
            <div style="font-family:'JetBrains Mono', monospace; font-size:11px; color:#CBD5E1;">
                {event_html_items}
            </div>
        </div>
        """)

    with col_news:
        live_news = fetch_live_market_news(limit=4)
        if not live_news:
            st.html("""
            <div class="monolith-card" style="padding:16px; height:100%;">
                <div style="font-family:'Syncopate', sans-serif; font-size:11px; font-weight:700; color:#FFFFFF; margin-bottom:12px;">
                    ⚡ IMPACT INTELLIGENCE & NEWS AUDIT
                </div>
                <div style="color:#94A3B8; font-size:12px; padding:28px 16px; text-align:center; font-family:'JetBrains Mono', monospace; background:rgba(0,0,0,0.3); border-radius:8px; border:1px dashed rgba(255,255,255,0.1);">
                    📡 News Feed Currently Unavailable<br>
                    <span style="font-size:10px; color:#64748B;">Official RSS feeds unreachable or network offline</span>
                </div>
            </div>
            """)
        else:
            news_html_items = ""
            for item in live_news:
                source = html.escape(str(item.get('source', 'Financial Wire')))
                credibility = html.escape(str(item.get('credibility', 'TIER 1 (OFFICIAL / VERIFIED)')))
                color = item.get('color', '#10B981')
                score = item.get('score', 8.0)
                title = html.escape(str(item.get('title', '')))
                expected_move = html.escape(str(item.get('expected_move', '100 - 200 pts')))
                tag = html.escape(str(item.get('tag', 'MARKET UPDATE')))
                link = html.escape(str(item.get('link', '#')))
                pub_time = html.escape(str(item.get('published_ist') or item.get('time', '')))

                news_html_items += f"""
                <div style="background:rgba(0,0,0,0.4); border-left:3px solid {color}; padding:9px 12px; border-radius:8px; margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; font-size:10px; font-family:'JetBrains Mono', monospace;">
                        <span style="color:#94A3B8;">{source} • {credibility} {f'• {pub_time}' if pub_time else ''}</span>
                        <b style="color:{color}; background:rgba(255,255,255,0.06); padding:2px 6px; border-radius:4px;">{score}/10 IMPACT</b>
                    </div>
                    <div style="font-size:12px; font-weight:600; color:#E2E8F0; margin:4px 0; line-height:1.3;">
                        <a href="{link}" target="_blank" rel="noopener noreferrer" style="color:#E2E8F0; text-decoration:none;">{title} ↗</a>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:10px; font-family:'JetBrains Mono', monospace; color:#64748B;">
                        <span>Expected Move: <b style="color:#FCD34D;">{expected_move}</b></span>
                        <span style="color:#94A3B8;">{tag}</span>
                    </div>
                </div>
                """
            st.html(f"""
            <div class="monolith-card" style="padding:16px; height:100%;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <div style="font-family:'Syncopate', sans-serif; font-size:11px; font-weight:700; color:#FFFFFF;">
                        ⚡ IMPACT INTELLIGENCE & NEWS AUDIT
                    </div>
                    <span style="font-size:10px; font-family:'JetBrains Mono', monospace; color:#10B981; background:rgba(16,185,129,0.1); padding:2px 8px; border-radius:12px; border:1px solid rgba(16,185,129,0.3);">TIER 1 RSS</span>
                </div>
                {news_html_items}
            </div>
            """)

def render_chartink_pro_suite(live_snap: Dict[str, Any]) -> None:
    """Renders Mode 2: Chartink Pro Suite with sortable dataframes and interactive candlestick/RSI."""
    # 0. Startup Guard: Backfill depth and staleness verification
    health = check_bhavcopy_backfill_health()
    if not health.get("is_healthy", False):
        st.html(f"""
        <div style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3); border-radius:12px; padding:24px; text-align:center; margin:20px 0;">
            <div style="font-family:'Syncopate', sans-serif; font-size:16px; font-weight:800; color:#EF4444; letter-spacing:1px; margin-bottom:8px;">
                ⚠️ DATA UNAVAILABLE / BACKFILL INCOMPLETE
            </div>
            <div style="font-family:'JetBrains Mono', monospace; font-size:12px; color:#FCA5A5;">
                {html.escape(health.get('status_message', 'Bhavcopy backfill has not reached minimum required depth.'))}
            </div>
            <div style="font-family:'JetBrains Mono', monospace; font-size:11px; color:#94A3B8; margin-top:8px;">
                Real Sessions: {health.get('num_sessions', 0)} / 250 required • Latest Session: {health.get('max_date', 'None')}
            </div>
        </div>
        """)
        return

    # 1. Navbar
    st.html("""
    <div style="background:#0B132B; border:1px solid #1C2541; border-radius:14px; padding:14px 18px; margin-bottom:18px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; box-shadow:0 10px 30px rgba(0,0,0,0.6);">
        <div style="display:flex; align-items:center; flex-wrap:wrap; gap:14px;">
            <div style="display:flex; align-items:center; gap:6px;">
                <span style="font-family:'Syncopate', sans-serif; font-size:clamp(18px, 3vw, 24px); font-weight:900; color:#FF6B00; letter-spacing:2px;">CHART<span style="color:#FFFFFF;">INK</span></span>
                <span style="background:rgba(255,107,0,0.18); color:#FF6B00; border:1px solid rgba(255,107,0,0.35); font-size:10px; font-weight:800; padding:2px 8px; border-radius:4px; font-family:'JetBrains Mono', monospace;">PRO REA CLONE</span>
            </div>
            <div style="height:20px; width:1px; background:rgba(255,255,255,0.1);"></div>
            <div style="display:flex; flex-wrap:wrap; gap:10px; font-size:11px; font-weight:600; color:#94A3B8;">
                <span style="color:#FFFFFF; cursor:pointer;">⚡ Screener</span>
                <span style="color:#94A3B8; cursor:pointer;">📊 Atlas Dashboard</span>
                <span style="color:#94A3B8; cursor:pointer;">🛠️ Custom Query</span>
                <span style="color:#94A3B8; cursor:pointer;">🕯️ Candlestick</span>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:12px;">
            <span style="font-family:'JetBrains Mono', monospace; font-size:10px; color:#10B981; background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3); padding:4px 12px; border-radius:20px;">
                ● NSE CASH UNIVERSE (EQ & BE Series • Full EOD History)
            </span>
        </div>
    </div>
    """)

    # 2. Market Breadth Health Gauge
    breadth = get_market_breadth_stats()
    as_of_badge = f"AS OF: {breadth.get('as_of_date', 'N/A')} • {breadth.get('total_universe', 0)} NSE EQUITIES"

    st.html(f"""
    <div style="background:#0F172A; border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:16px; margin-bottom:18px;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:12px; padding-bottom:8px; border-bottom:1px solid rgba(255,255,255,0.06);">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-size:16px;">📊</span>
                <b style="font-family:'Syncopate', sans-serif; font-size:11px; color:#FFFFFF; letter-spacing:1px;">CHARTINK ATLAS: WHOLE UNIVERSE MARKET BREADTH</b>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-family:'JetBrains Mono', monospace; font-size:10px; color:#94A3B8; background:rgba(255,255,255,0.06); padding:3px 10px; border-radius:12px; border:1px solid rgba(255,255,255,0.1);">
                    📅 {html.escape(as_of_badge)}
                </span>
                <span style="font-family:'JetBrains Mono', monospace; font-size:10px; font-weight:800; color:{breadth.get('regime_color', '#10B981')}; background:{breadth.get('regime_color', '#10B981')}22; padding:3px 10px; border-radius:12px; border:1px solid {breadth.get('regime_color', '#10B981')}55;">
                    ● {html.escape(breadth.get('market_regime', 'HEALTHY EXPANSION'))}
                </span>
            </div>
        </div>
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap:10px; font-family:'JetBrains Mono', monospace;">
            <div style="background:rgba(0,0,0,0.4); padding:12px; border-radius:10px; border:1px solid rgba(255,255,255,0.05);">
                <div style="font-size:9.5px; color:#94A3B8;">STOCKS > 20 EMA</div>
                <div style="font-size:20px; font-weight:900; color:#10B981; margin:2px 0;">{breadth.get('pct_above_20_ema', 0.0)}%</div>
                <div style="font-size:9.5px; color:#64748B;">{breadth.get('above_20', 0)}/{breadth.get('eligible_20_count', 0)} eligible (≥20d)</div>
            </div>
            <div style="background:rgba(0,0,0,0.4); padding:12px; border-radius:10px; border:1px solid rgba(255,255,255,0.05);">
                <div style="font-size:9.5px; color:#94A3B8;">STOCKS > 50 EMA</div>
                <div style="font-size:20px; font-weight:900; color:#00F0FF; margin:2px 0;">{breadth.get('pct_above_50_ema', 0.0)}%</div>
                <div style="font-size:9.5px; color:#64748B;">{breadth.get('above_50', 0)}/{breadth.get('eligible_50_count', 0)} eligible (≥50d)</div>
            </div>
            <div style="background:rgba(0,0,0,0.4); padding:12px; border-radius:10px; border:1px solid rgba(255,255,255,0.05);">
                <div style="font-size:9.5px; color:#94A3B8;">STOCKS > 200 EMA</div>
                <div style="font-size:20px; font-weight:900; color:#F59E0B; margin:2px 0;">{breadth.get('pct_above_200_ema', 0.0)}%</div>
                <div style="font-size:9.5px; color:#64748B;">{breadth.get('above_200', 0)}/{breadth.get('eligible_200_count', 0)} eligible (≥200d)</div>
            </div>
            <div style="background:rgba(0,0,0,0.4); padding:12px; border-radius:10px; border:1px solid rgba(255,255,255,0.05);">
                <div style="font-size:9.5px; color:#94A3B8;">ADVANCE / DECLINE</div>
                <div style="font-size:20px; font-weight:900; color:#10B981; margin:2px 0;">{html.escape(str(breadth.get('adv_dec_ratio', '1.0'))[:6])}</div>
                <div style="font-size:9.5px; color:#10B981;">{breadth.get('advances', 0)} Adv / {breadth.get('declines', 0)} Dec</div>
            </div>
        </div>
    </div>
    """)

    # 3. Dynamic Sector Heatmap
    sectors = get_sector_heatmap()
    sec_html_cards = ""
    for sc in sectors:
        border_c = sc.get('color', '#10B981')
        sec_name = html.escape(str(sc.get('sector', '')))
        sec_chg = html.escape(str(sc.get('chg', '')))
        sec_status = html.escape(str(sc.get('status', '')))
        sec_html_cards += f"""
        <div style="background:#0F172A; border-top:3px solid {border_c}; padding:10px; border-radius:8px; text-align:center; font-family:'JetBrains Mono', monospace;">
            <div style="font-size:10px; color:#94A3B8;">{sec_name}</div>
            <div style="font-size:16px; font-weight:900; color:{border_c}; margin:2px 0;">{sec_chg}</div>
            <div style="font-size:9px; color:#CBD5E1;">{sec_status}</div>
        </div>
        """

    st.html(f"""
    <div class="sector-responsive-grid">
        {sec_html_cards}
    </div>
    """)

    # 4. Scanner Category Selector & Filter Clauses
    scanner_names = list(CHARTINK_CATEGORIES.keys())
    sel_cat = st.selectbox("Select Dynamic Technical Screener:", scanner_names, index=0)
    active_cat = CHARTINK_CATEGORIES.get(sel_cat, {})
    clauses_html = "<br>".join([f"<span style='color:#E2E8F0;'>{html.escape(c)}</span>" for c in active_cat.get('clauses', [])])

    st.html(f"""
    <div style="background:#0B132B; border:1px solid rgba(255,107,0,0.3); border-radius:12px; padding:16px; margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="color:#FF6B00; font-size:16px;">🛠️</span>
                <b style="color:#FFFFFF; font-family:'Syncopate', sans-serif; font-size:11px;">CHARTINK FILTER QUERY: {html.escape(sel_cat)}</b>
            </div>
        </div>
        <div style="font-family:'JetBrains Mono', monospace; font-size:12px; line-height:1.6; background:rgba(0,0,0,0.5); padding:12px; border-radius:8px; border:1px solid rgba(255,255,255,0.05);">
            {clauses_html}
        </div>
        <div style="font-size:11px; color:#94A3B8; margin-top:8px;">
            <b>Scan Logic:</b> {html.escape(active_cat.get('desc', ''))}
        </div>
        <div style="font-size:11px; color:#FCD34D; margin-top:8px; font-family:'JetBrains Mono', monospace; background:rgba(245,158,11,0.08); padding:6px 10px; border-radius:6px; border:1px solid rgba(245,158,11,0.2);">
            ⚠️ <b>Screening Note:</b> {html.escape(active_cat.get('note', 'Screening filter based on historical EOD technical criteria; does not constitute a directional prediction or trading signal.'))}
        </div>
    </div>
    """)

    # Run Scan Execution Button & Session State Persistence
    col_run_btn, col_run_info = st.columns([1.2, 3])
    with col_run_btn:
        run_scan_clicked = st.button("⚡ RUN SCAN", type="primary", use_container_width=True)

    # Check if scan needs execution
    should_run = (
        run_scan_clicked or
        ("chartink_scan_data" not in st.session_state) or
        (st.session_state.get("chartink_last_scan_name") != sel_cat)
    )

    if should_run:
        from chartink_engine import run_chartink_scan
        with st.spinner(f"Evaluating {sel_cat} across whole NSE equity universe in DuckDB..."):
            scan_df, scan_meta = run_chartink_scan(sel_cat)
            st.session_state["chartink_scan_data"] = scan_df
            st.session_state["chartink_scan_meta"] = scan_meta
            st.session_state["chartink_last_scan_name"] = sel_cat

    scan_df = st.session_state.get("chartink_scan_data", pd.DataFrame())
    scan_meta = st.session_state.get("chartink_scan_meta", {})

    with col_run_info:
        as_of_info = scan_meta.get("as_of_date", breadth.get("as_of_date", "N/A"))
        total_syms = scan_meta.get("total_universe", breadth.get("total_universe", 0))
        matched_syms = len(scan_df)
        st.html(f"""
        <div style="font-family:'JetBrains Mono', monospace; font-size:11px; color:#94A3B8; padding:8px 12px; background:rgba(0,0,0,0.3); border-radius:8px; border:1px solid rgba(255,255,255,0.06);">
            🎯 <b>{matched_syms}</b> stocks matched out of <b>{total_syms}</b> active NSE equities • <span style="color:#FCD34D;">As of {as_of_info}</span>
        </div>
        """)

    # 5. Clean Sortable Results Table (ZERO hardcoded targets/stop-loss/conviction)
    st.html(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin:14px 0 8px 0;">
        <span style="font-family:'Syncopate', sans-serif; font-size:11px; font-weight:700; color:#FFFFFF;">
            📋 LIVE SCANNER RESULTS: {html.escape(sel_cat)}
        </span>
        <span style="font-family:'JetBrains Mono', monospace; font-size:10px; color:#10B981;">
            ● NATIVE SORTABLE • ZERO SYNTHETIC PREDICTIONS
        </span>
    </div>
    """)

    if scan_df.empty:
        st.info(f"No stocks currently match the filter criteria for '{sel_cat}' as of {as_of_info}.")
    else:
        st.dataframe(
            scan_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Sr": st.column_config.NumberColumn("Sr", width="small"),
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Price": st.column_config.NumberColumn("Price (₹)", format="₹%.2f"),
                "Change %": st.column_config.NumberColumn("Change %", format="%.2f%%"),
                "Volume": st.column_config.NumberColumn("Volume", format="%d"),
                "Volume Surge": st.column_config.TextColumn("Volume Surge", width="small"),
                "RSI": st.column_config.NumberColumn("RSI (14)", format="%.1f"),
                "EMA 20": st.column_config.NumberColumn("EMA 20 (₹)", format="₹%.2f"),
                "EMA 50": st.column_config.NumberColumn("EMA 50 (₹)", format="₹%.2f"),
                "EMA 200": st.column_config.NumberColumn("EMA 200 (₹)", format="₹%.2f"),
                "ATR": st.column_config.NumberColumn("ATR (14) (₹)", format="₹%.2f")
            }
        )

    # 6. Interactive Candlestick & RSI Inspector
    st.html("""
    <div class="section-header"><span>02 / CHARTINK INTERACTIVE CHART & RSI(14) INSPECTOR</span></div>
    """)

    available_symbols = scan_df["Symbol"].tolist() if not scan_df.empty else ["RELIANCE", "HDFCBANK", "SBIN", "TRENT"]
    selected_inspect_stock = st.selectbox("Select Stock for Candlestick Inspector:", available_symbols, index=0)
    inspect_df = fetch_stock_real_ohlc(selected_inspect_stock, period="3mo")
    is_genuine_data = inspect_df.get("is_real", pd.Series([True])).iloc[0] if 'is_real' in inspect_df.columns else True
    data_badge = "🟢 GENUINE DAILY OHLC (NSE)" if is_genuine_data else "🟡 OFFLINE CACHED OHLC"

    st.html(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; font-family:'JetBrains Mono', monospace; font-size:11px;">
        <span style="color:#94A3B8;">Asset: <b style="color:#FFFFFF;">{html.escape(selected_inspect_stock)}</b></span>
        <span style="color:#10B981; background:rgba(16,185,129,0.12); padding:3px 10px; border-radius:6px; border:1px solid rgba(16,185,129,0.3);">{data_badge}</span>
    </div>
    """)

    fig_chartink = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.72, 0.28]
    )

    fig_chartink.add_trace(go.Candlestick(
        x=inspect_df['Date'], open=inspect_df['Open'], high=inspect_df['High'],
        low=inspect_df['Low'], close=inspect_df['Close'],
        name=selected_inspect_stock,
        increasing_line_color='#10B981', decreasing_line_color='#F43F5E'
    ), row=1, col=1)

    fig_chartink.add_trace(go.Scatter(x=inspect_df['Date'], y=inspect_df['EMA20'], line=dict(color='#00F0FF', width=1.5), name='20 EMA'), row=1, col=1)
    fig_chartink.add_trace(go.Scatter(x=inspect_df['Date'], y=inspect_df['EMA50'], line=dict(color='#F59E0B', width=1.5), name='50 EMA'), row=1, col=1)
    fig_chartink.add_trace(go.Scatter(x=inspect_df['Date'], y=inspect_df['RSI'], line=dict(color='#FCD34D', width=1.5), name='RSI(14)'), row=2, col=1)
    fig_chartink.add_hline(y=70, line_dash="dash", line_color="#F43F5E", row=2, col=1)
    fig_chartink.add_hline(y=30, line_dash="dash", line_color="#10B981", row=2, col=1)

    fig_chartink.update_layout(
        template="plotly_dark",
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_rangeslider_visible=False,
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_chartink, use_container_width=True)

