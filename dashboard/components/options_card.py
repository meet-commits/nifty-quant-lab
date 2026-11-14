"""
dashboard/components/options_card.py
Honest Option Chain Status Card.
Replaces circular synthetic options profile with an explicit disconnected state.
"""

import streamlit as st

def render_options_card() -> None:
    """Renders honest Option Chain Disconnected card."""
    st.html("<div class='section-header'><span>03 / DERIVATIVES & OPTIONS CHAIN STATUS</span></div>")

    st.html("""
    <div class="monolith-card" style="padding:22px; border-left:4px solid #64748B; margin-bottom:18px;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:12px;">
            <div style="display:flex; align-items:center; gap:12px;">
                <span style="font-size:24px;">🔌</span>
                <div>
                    <div style="font-family:'Syncopate', sans-serif; font-size:12px; font-weight:700; color:#FFFFFF; letter-spacing:2px;">
                        LIVE OPTION CHAIN (OI / PCR / MAX PAIN)
                    </div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:10px; color:#94A3B8; letter-spacing:1px; margin-top:2px;">
                        TICK-BY-TICK EXCHANGE DERIVATIVES FEED
                    </div>
                </div>
            </div>
            <span style="font-family:'JetBrains Mono', monospace; font-size:11px; font-weight:800; color:#94A3B8; background:rgba(148,163,184,0.12); border:1px solid rgba(148,163,184,0.3); padding:5px 14px; border-radius:999px;">
                ⚪ STATUS: NOT CONNECTED
            </span>
        </div>

        <div style="font-size:13px; color:#CBD5E1; line-height:1.6; margin-bottom:12px;">
            Live strike-by-strike open interest (Put/Call OI, dynamic Put-Call Ratio, and Max Pain) requires an active authenticated broker API connection (e.g., Zerodha Kite Connect, Upstox, or Angel One SmartAPI).
        </div>

        <div style="background:rgba(0,0,0,0.4); padding:12px 16px; border-radius:10px; border:1px solid rgba(255,255,255,0.06); font-family:'JetBrains Mono', monospace; font-size:11px; color:#94A3B8;">
            🛡️ <b>Truth-in-Data Protocol:</b> In accordance with quantitative integrity standards, simulated/circular option distribution models (which previously approximated OI shifts based on spot price) have been completely decommissioned. Connect your broker API token in <code>config/broker.yaml</code> to enable live tick-by-tick option metrics.
        </div>
    </div>
    """)

