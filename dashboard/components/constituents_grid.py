"""
dashboard/components/constituents_grid.py
Index-Specific Constituents Component.
Displays constituents driven strictly by config/weights.csv with 'as of' date:
  - Bank Nifty uses pure banking stocks only
  - Sensex uses BSE-30 constituents
  - Nifty 50 uses Nifty-50 heavyweights
"""

import html
import streamlit as st
from typing import Dict, Any

def render_constituents_grid(selected_asset: str, live_snap: Dict[str, Any]) -> None:
    """Renders index-specific constituents cards loaded from config/weights.csv."""
    constituents_map = live_snap.get("constituents_by_index", {})

    # Select appropriate index constituent list
    if selected_asset == "BANK NIFTY":
        idx_key = "BANK NIFTY"
        stocks = constituents_map.get("BANK NIFTY", [])
    elif selected_asset == "SENSEX":
        idx_key = "SENSEX"
        stocks = constituents_map.get("SENSEX", [])
    else:
        idx_key = "NIFTY 50"
        stocks = constituents_map.get("NIFTY 50") or live_snap.get("heavyweights", [])

    as_of_date = stocks[0].get("as_of_date", "Latest") if stocks else "Latest"

    st.html(f"""
    <div class='section-header'>
        <span>02 / {html.escape(idx_key)} CONSTITUENT DRIVERS (WEIGHTS AS OF {html.escape(as_of_date)})</span>
    </div>
    """)

    if not stocks:
        st.info(f"Constituent data for {html.escape(idx_key)} is currently offline.")
        return

    hw_html_cards = ""
    for s in stocks:
        is_bull = s.get('bull', False)
        border_c = "#10B981" if is_bull else "#F43F5E"
        sym_clean = html.escape(str(s.get('sym', '')))
        wt_clean = html.escape(str(s.get('wt', '')))
        price_clean = html.escape(str(s.get('price', 'N/A')))
        pct_clean = html.escape(str(s.get('pct', 'N/A')))
        status_clean = html.escape(str(s.get('status', '')))

        hw_html_cards += f"""
        <div class="monolith-card" style="border:1px solid {border_c}; padding:14px; margin-bottom:0;">
            <div style="display:flex; justify-content:space-between; font-size:11px; color:#94A3B8;">
                <b style="color:#FFFFFF; font-size:13px; font-family:'JetBrains Mono', monospace;">{sym_clean}</b>
                <span title="Official index weighting as of {html.escape(as_of_date)}" style="cursor:help;">{wt_clean} Wt</span>
            </div>
            <div style="font-size:19px; font-weight:900; font-family:'JetBrains Mono', monospace; color:#FFFFFF; margin:5px 0;">{price_clean}</div>
            <div style="font-size:12px; font-weight:800; color:{border_c};">{pct_clean}</div>
            <div style="font-size:11px; color:#CBD5E1; margin-top:4px;">{status_clean}</div>
        </div>
        """

    st.html(f"""
    <div class="heavyweights-responsive-grid">
        {hw_html_cards}
    </div>
    """)

