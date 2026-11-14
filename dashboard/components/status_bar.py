"""
dashboard/components/status_bar.py
Top status bar component with balanced HTML tags, live MCP badge, and session indicator.
"""

import html
import streamlit as st
from typing import Dict, Any

def render_status_bar(live_snap: Dict[str, Any], latest_date: str) -> None:
    """Renders top header status bar with balanced HTML tags."""
    if not live_snap or not isinstance(live_snap, dict):
        live_snap = {}

    status_color = live_snap.get("status_color", "#64748B")
    session_status = live_snap.get("session_status", "DATA UNAVAILABLE")
    mcp_info = live_snap.get("mcp_health") or {}
    is_mcp_online = mcp_info.get("is_online", True)

    mcp_badge_html = """
    <div style="font-family:'JetBrains Mono', monospace; font-size:10.5px; color:#00F0FF; font-weight:700; background:rgba(0,240,255,0.10); border:1px solid rgba(0,240,255,0.3); padding:5px 14px; border-radius:999px; display:flex; align-items:center; gap:6px;">
        <span style="height:7px; width:7px; border-radius:999px; background:#00F0FF; box-shadow:0 0 8px #00F0FF;"></span>
        OFFICIAL NSE MCP: ACTIVE (mcp.nseindia.in)
    </div>
    """ if is_mcp_online else """
    <div style="font-family:'JetBrains Mono', monospace; font-size:10.5px; color:#F59E0B; font-weight:700; background:rgba(245,158,11,0.10); border:1px solid rgba(245,158,11,0.3); padding:5px 14px; border-radius:999px; display:flex; align-items:center; gap:6px;">
        <span style="height:7px; width:7px; border-radius:999px; background:#F59E0B;"></span>
        MCP STANDBY / FALLBACK ACTIVE
    </div>
    """

    c_bar_l, c_bar_r = st.columns([3, 1])
    with c_bar_l:
        st.html(f"""
        <div class="monolith-card" style="padding:14px 22px; margin-bottom:14px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
            <div style="display:flex; align-items:center; gap:16px;">
                <div style="height:40px; width:40px; border-radius:12px; background:linear-gradient(135deg, #FFF5C0, #F59E0B); display:flex; align-items:center; justify-content:center; color:#000000; font-size:20px; font-weight:900; box-shadow:0 0 25px rgba(245,158,11,0.45);">
                    ◆
                </div>
                <div>
                    <div style="font-family:'Syncopate', 'Clash Display', sans-serif; font-size:17px; font-weight:700; letter-spacing:4px; color:#FFFFFF;">
                        APEX <span class="gold-gradient-text">QUANT 999</span>
                    </div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:10px; color:#94A3B8; letter-spacing:1px; margin-top:2px;">
                        LIVE SUB-10S TICK PIPELINE • ASIA/KOLKATA TIMEZONE
                    </div>
                </div>
            </div>
            <div style="display:flex; align-items:center; flex-wrap:wrap; gap:10px;">
                {mcp_badge_html}
                <div style="font-family:'JetBrains Mono', monospace; font-size:11px; color:{status_color}; font-weight:700; background:rgba(16,185,129,0.12); border:1px solid {status_color}55; padding:6px 16px; border-radius:999px; display:flex; align-items:center; gap:8px;">
                    <span style="height:8px; width:8px; border-radius:999px; background:{status_color}; box-shadow:0 0 10px {status_color};"></span>
                    {html.escape(session_status)} • {html.escape(latest_date)}
                </div>
            </div>
        </div>
        """)
    with c_bar_r:
        if st.button("🔄 Force Refresh Ticks Now", use_container_width=True):
            st.rerun()

