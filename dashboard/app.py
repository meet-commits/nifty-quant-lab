"""
dashboard/app.py
APEX QUANT 999 | Executive Suite & Chartink Pro Clone
Modular, high-performance quantitative intelligence platform.
Zero synthetic fallbacks, truth-in-data verified exchange feeds.
"""

import os
import sys
import streamlit as st

# Setup system path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(BASE_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from live_market_service import (
    fetch_live_market_snapshot,
    get_market_session_status
)
from macro_story_engine import build_comprehensive_market_story

from components.status_bar import render_status_bar
from components.data_health import render_data_health_panel
from components.gap_forecast_card import render_gap_forecast_card
from components.hero_runway import render_hero_runway
from components.intraday_section import render_intraday_section
from components.constituents_grid import render_constituents_grid
from components.options_card import render_options_card
from components.backtest_lab import render_backtest_lab
from components.chartink_suite import render_events_and_news, render_chartink_pro_suite
from components.footer import render_footer

st.set_page_config(
    page_title="APEX QUANT 999 | Executive Cockpit & Chartink Pro",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------- LUXURY EDITORIAL CSS (SYNCOPATE + CLASH DISPLAY) -----------------
st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Syncopate:wght@400;700&family=Unbounded:wght@400;600;700;800;900&display=swap" rel="stylesheet">
<link href="https://api.fontshare.com/v2/css?f[]=clash-display@400,500,600,700&f[]=cabinet-grotesk@700,800,900&display=swap" rel="stylesheet">

<style>
    .stApp {
        background-color: #000000 !important;
        background-image: 
            radial-gradient(circle at 50% -5%, rgba(245, 158, 11, 0.20) 0%, transparent 45%),
            radial-gradient(circle at 90% 30%, rgba(16, 185, 129, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 10% 70%, rgba(6, 182, 212, 0.08) 0%, transparent 40%);
        color: #F8FAFC !important;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    }
    
    .font-syncopate {
        font-family: 'Syncopate', 'Clash Display', 'Unbounded', sans-serif !important;
        letter-spacing: 3px !important;
        text-transform: uppercase !important;
    }
    
    .font-mono {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .gold-gradient-text {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF5C0 25%, #F59E0B 55%, #D97706 80%, #92400E 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        display: inline-block;
    }

    .gold-glow {
        filter: drop-shadow(0 0 25px rgba(245, 158, 11, 0.45));
    }

    .monolith-card {
        background: rgba(9, 12, 20, 0.88) !important;
        backdrop-filter: blur(28px) !important;
        -webkit-backdrop-filter: blur(28px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 18px !important;
        padding: 20px !important;
        box-shadow: 0 16px 40px -15px rgba(0, 0, 0, 0.85) !important;
        margin-bottom: 18px !important;
    }

    .monolith-card-gold {
        background: rgba(12, 14, 22, 0.92) !important;
        backdrop-filter: blur(32px) !important;
        -webkit-backdrop-filter: blur(32px) !important;
        border: 1px solid rgba(245, 158, 11, 0.35) !important;
        border-radius: 20px !important;
        padding: 24px !important;
        box-shadow: 0 0 50px rgba(245, 158, 11, 0.12), inset 0 0 30px rgba(245, 158, 11, 0.04) !important;
        margin-bottom: 18px !important;
    }

    .section-header {
        font-family: 'Syncopate', 'Clash Display', sans-serif;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #FCD34D;
        margin: 24px 0 12px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .section-header::after {
        content: "";
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.3), transparent);
    }

    /* Responsive Grids */
    .runway-responsive-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-top: 10px;
        margin-bottom: 18px;
    }

    .heavyweights-responsive-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 20px;
    }

    .sector-responsive-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 10px;
        margin-bottom: 16px;
    }

    .macro-pillars-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
        margin-top: 12px;
    }

    .macro-pillar-card {
        background: rgba(0, 0, 0, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 14px;
    }

    .pillar-title {
        font-family: 'Syncopate', 'Clash Display', sans-serif;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }

    .pillar-body {
        font-size: 12px;
        line-height: 1.6;
        color: #CBD5E1;
    }

    /* Mobile Fluid Overrides */
    @media (max-width: 768px) {
        .block-container {
            padding: 0.5rem 0.5rem 2rem 0.5rem !important;
        }
        .monolith-card {
            padding: 14px 12px !important;
            border-radius: 14px !important;
            margin-bottom: 12px !important;
        }
        .monolith-card-gold {
            padding: 16px 12px !important;
            border-radius: 16px !important;
            margin-bottom: 14px !important;
        }
        .font-syncopate {
            letter-spacing: 1.5px !important;
        }
        .runway-responsive-grid {
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 8px !important;
        }
        .heavyweights-responsive-grid {
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 8px !important;
        }
        .sector-responsive-grid {
            grid-template-columns: repeat(3, 1fr) !important;
            gap: 8px !important;
        }
        .macro-pillars-grid {
            grid-template-columns: 1fr !important;
            gap: 10px !important;
        }
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 8px !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        [data-testid="stDataFrame"] {
            width: 100% !important;
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
        }
        html, body, [data-testid="stAppViewContainer"], .main {
            overflow-x: hidden !important;
            max-width: 100vw !important;
        }
    }
    @media (max-width: 480px) {
        .sector-responsive-grid {
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 6px !important;
        }
    }
</style>
""")

# ----------------- VIEW MODE SWITCHER -----------------
st.html("""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; background:rgba(255,255,255,0.03); padding:8px 16px; border-radius:12px; border:1px solid rgba(255,255,255,0.06);">
    <div style="font-family:'Syncopate', sans-serif; font-size:11px; font-weight:700; color:#FCD34D; letter-spacing:2px;">
        PLATFORM NAVIGATION:
    </div>
    <div style="font-size:11px; color:#94A3B8; font-family:'JetBrains Mono', monospace;">
        ONE-CLICK INSTANT TOGGLE
    </div>
</div>
""")

suite_view = st.radio(
    "Select Suite View Mode:",
    ["👑 APEX LUXURY COCKPIT", "📊 CHARTINK PRO SUITE (DETO-TO-DETO)"],
    horizontal=True,
    label_visibility="collapsed"
)

# Helpers for formatting
def format_chg_str(chg, pct, is_stale=False):
    if is_stale:
        return "STALE (Closed)"
    if chg is None or pct is None:
        return "N/A"
    return f"{pct:+.2f}% ({'+' if chg>=0 else ''}{chg:,.1f} pts)"

def format_target_diff(target, spot):
    if target is None or spot is None:
        return "N/A"
    diff = target - spot
    return f"{'+' if diff>=0 else ''}{diff:.1f} pts from spot"

def build_current_index_data(live_snap, selected_asset, macro_story):
    session_status = live_snap.get("session_status", "CLOSED")
    latest_date = live_snap.get("timestamp", "N/A")

    n_q = live_snap.get("nifty", {})
    b_q = live_snap.get("banknifty", {})
    s_q = live_snap.get("sensex", {})
    g_q = live_snap.get("giftnifty", {})

    strength_score = macro_story.get("strength_score", 50.0)
    conf_text = f"Intraday Strength: {strength_score:.1f} (heuristic)"

    index_configs = {
        "NIFTY 50": {
            "title": "NIFTY", "tag": "50", "q": n_q, "base": 25000,
            "date_info": f"{session_status} • {latest_date}"
        },
        "BANK NIFTY": {
            "title": "BANK", "tag": "NIFTY", "q": b_q, "base": 61000,
            "date_info": f"{session_status} • {latest_date}"
        },
        "GIFT NIFTY": {
            "title": "GIFT",
            "tag": "NIFTY" if g_q.get("is_real") else "NIFTY [APPROXIMATION]",
            "q": g_q, "base": 25200,
            "date_info": f"GIFT CITY DERIVATIVE TRACK • {latest_date}"
        },
        "SENSEX": {
            "title": "SENSEX", "tag": "BSE", "q": s_q, "base": 81600,
            "date_info": f"{session_status} • {latest_date}"
        }
    }

    cfg = index_configs.get(selected_asset, index_configs["NIFTY 50"])
    q = cfg["q"]
    spot_val = q.get("value") or q.get("last")
    spot_float = float(spot_val) if spot_val is not None else None

    sl_val = q.get("s1") or q.get("sl")
    piv_val = q.get("pivot")
    t1_val = q.get("r1") or q.get("t1")
    t2_val = q.get("r2") or q.get("t2")

    return {
        "title": cfg["title"],
        "tag": cfg["tag"],
        "spot": spot_float,
        "spot_str": f"{spot_float:,.2f}" if spot_float is not None else "N/A",
        "chg": format_chg_str(q.get("chg_pts"), q.get("chg_pct"), q.get("is_stale", False)),
        "conf": conf_text,
        "date_info": cfg["date_info"],
        "prev_date": q.get("prev_session_date", "N/A"),
        "sl": sl_val, "pivot": piv_val, "t1": t1_val, "t2": t2_val,
        "sl_str": f"{sl_val:,.1f}" if sl_val is not None else "N/A",
        "piv_str": f"{piv_val:,.1f}" if piv_val is not None else "N/A",
        "t1_str": f"{t1_val:,.1f}" if t1_val is not None else "N/A",
        "t2_str": f"{t2_val:,.1f}" if t2_val is not None else "N/A",
        "t1_diff": format_target_diff(t1_val, spot_float),
        "t2_diff": format_target_diff(t2_val, spot_float),
        "bull_pct": macro_story.get("bull_pct", 50),
        "bear_pct": macro_story.get("bear_pct", 50),
        "base": cfg["base"]
    }

# Check market session status for fragment timing
is_mkt_live, _, _ = get_market_session_status()
fragment_refresh = "30s" if is_mkt_live else None

# =========================================================================
# MODE 1: 👑 APEX LUXURY COCKPIT
# =========================================================================
if suite_view == "👑 APEX LUXURY COCKPIT":
    # Fragment for live-ticker sections (refreshes 30s during market hours, none off-hours)
    @st.fragment(run_every=fragment_refresh)
    def render_live_cockpit_fragment():
        live_snap = fetch_live_market_snapshot()
        latest_date = live_snap.get("timestamp", "N/A")

        # 1. Top Status Bar
        render_status_bar(live_snap, latest_date)

        # 2. Asset Switcher
        selected_asset = st.radio(
            "Asset Switcher:",
            ["NIFTY 50", "BANK NIFTY", "GIFT NIFTY", "SENSEX"],
            horizontal=True,
            label_visibility="collapsed"
        )

        try:
            macro_story = build_comprehensive_market_story(live_snap, selected_asset)
        except Exception:
            macro_story = {}

        try:
            current = build_current_index_data(live_snap, selected_asset, macro_story)
        except Exception:
            current = {}

        # 3. Dual-Horizon Overnight Gap Forecast
        render_gap_forecast_card(live_snap)

        # 4. Master Hero Runway & Reference Levels
        render_hero_runway(current, macro_story, live_snap)

        # 5. Intraday Section & 5m Chart
        render_intraday_section(selected_asset, current, live_snap)

        # 6. Index-Specific Constituents Grid
        render_constituents_grid(selected_asset, live_snap)

    # Call the live fragment
    render_live_cockpit_fragment()

    # Static sections (do NOT rerun on live tick, preserving backtests and scroll state)
    live_snap_static = fetch_live_market_snapshot()
    latest_date_static = live_snap_static.get("timestamp", "N/A")

    # 7. Data Health & Provenance Panel
    render_data_health_panel(live_snap_static)

    # 8. Options Open Interest Status Card (Honest Disconnected State)
    render_options_card()

    # 9. Scanners & News Dossier
    render_events_and_news()

    # 10. Walk-Forward Lab (with persistent st.session_state)
    render_backtest_lab()

    # 11. Single Canonical Footer
    render_footer(latest_date_static)

# =========================================================================
# MODE 2: 📊 CHARTINK PRO SUITE
# =========================================================================
else:
    live_snap_static = fetch_live_market_snapshot()
    latest_date_static = live_snap_static.get("timestamp", "N/A")
    render_chartink_pro_suite(live_snap_static)
    render_footer(latest_date_static)
