"""
dashboard/components/footer.py
Unified single footer line with statutory educational disclaimer.
"""

import html
import streamlit as st

def render_footer(latest_date: str) -> None:
    """Renders single canonical footer notice."""
    date_clean = html.escape(str(latest_date))
    st.html(f"""
    <div style="border-top:1px solid rgba(255,255,255,0.06); padding-top:18px; margin-top:28px; display:flex; justify-content:space-between; flex-wrap:wrap; gap:10px; font-family:'JetBrains Mono', monospace; font-size:11px; color:#64748B;">
        <span>Connected to Official NSE MCP Server (mcp.nseindia.in) • Local DuckDB Archive • Public Exchange Feeds (Latest: {date_clean})</span>
        <span>⚠️ Strictly for Quantitative Research & Education • Not SEBI-Registered Financial Advice</span>
    </div>
    """)

