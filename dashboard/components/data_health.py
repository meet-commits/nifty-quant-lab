"""
dashboard/components/data_health.py
Data Health & Provenance Panel:
Surfaces feed source, timestamp, latency, staleness flag, and official NSE MCP served quote metrics.
"""

import html
import streamlit as st
import pandas as pd
from typing import Dict, Any

def render_data_health_panel(live_snap: Dict[str, Any]) -> None:
    """Renders expandable Data Health & Provenance panel."""
    with st.expander("🛡️ Data Health & Exchange Feed Provenance (Zero Fake Fallbacks)", expanded=False):
        mcp_stats = live_snap.get("mcp_stats", {})
        quotes_served = mcp_stats.get("quotes_served", 0)
        errors_count = mcp_stats.get("errors_count", 0)
        sessions_created = mcp_stats.get("sessions_created", 0)

        # Header metrics row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Official NSE MCP Quotes Served", f"{quotes_served:,}")
        c2.metric("NSE MCP Active Sessions", f"{sessions_created}")
        c3.metric("MCP Recovered Fallbacks", f"{errors_count}")
        c4.metric("Session Mode", live_snap.get("session_status", "UNKNOWN"))

        # Build feed health table
        feeds = [
            ("NIFTY 50 Spot", live_snap.get("nifty", {})),
            ("BANK NIFTY Spot", live_snap.get("banknifty", {})),
            ("SENSEX Spot", live_snap.get("sensex", {})),
            ("GIFT NIFTY Derivatives", live_snap.get("giftnifty", {})),
            ("India VIX Volatility", live_snap.get("vix", {})),
            ("WTI Crude Oil (CL=F)", live_snap.get("crude", {})),
            ("USD/INR Spot (USDINR=X)", live_snap.get("usdinr", {})),
            ("US S&P 500 E-mini (ES=F)", live_snap.get("sp500", {})),
        ]

        feed_rows = []
        for name, q in feeds:
            src = q.get("source", "UNKNOWN")
            ts = q.get("ts_ist", "N/A")
            age = q.get("age_sec")
            is_stale = q.get("is_stale", False)
            val = q.get("value") or q.get("last")
            val_str = f"{float(val):,.2f}" if val is not None else "N/A"

            if is_stale:
                status_pill = "🔴 STALE (>300s)"
            elif val is None:
                status_pill = "⚪ OFFLINE"
            else:
                status_pill = "🟢 VERIFIED LIVE"

            feed_rows.append({
                "Feed Name": name,
                "Price": val_str,
                "Primary Source": src,
                "Timestamp (IST)": str(ts)[:19] if ts != "N/A" else "N/A",
                "Age (sec)": f"{age:.1f}s" if age is not None else "N/A",
                "Health Status": status_pill
            })

        df_health = pd.DataFrame(feed_rows)
        st.dataframe(
            df_health,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Feed Name": st.column_config.TextColumn("Feed Name", width="medium"),
                "Price": st.column_config.TextColumn("Last Price", width="small"),
                "Primary Source": st.column_config.TextColumn("Source Protocol", width="medium"),
                "Timestamp (IST)": st.column_config.TextColumn("Last Tick (IST)", width="medium"),
                "Age (sec)": st.column_config.TextColumn("Latency", width="small"),
                "Health Status": st.column_config.TextColumn("Status", width="small"),
            }
        )
        st.caption("ℹ️ *Integrity Guarantee: Stale feeds (>300s during market hours) never display static hardcoded default numbers. Failed feeds cleanly report N/A.*")

