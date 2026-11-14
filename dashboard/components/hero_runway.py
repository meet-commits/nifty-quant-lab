"""
dashboard/components/hero_runway.py
Hero Runway and Reference Level Cards.
Uses neutral quantitative reference levels:
  - Reference S1 (Prior session floor)
  - Pivot (P) (Central mean level)
  - Reference R1 (+X pts)
  - Reference R2 (+Y pts)
"""

import html
import streamlit as st
from typing import Dict, Any

def render_hero_runway(current: Dict[str, Any], macro_story: Dict[str, Any], live_snap: Dict[str, Any]) -> None:
    """Renders the Institutional Dossier, Master Hero card, and 4 Reference Level Cards."""

    # 1. Human Market Story Dossier
    regime_color = macro_story.get("regime_color", "#10B981")
    regime_badge = macro_story.get("regime_badge", "NEUTRAL REGIME")
    headline = macro_story.get("headline", "")
    triggers = macro_story.get("triggers", "")
    vix_status = macro_story.get("vix_status", "")
    vix_explanation = macro_story.get("vix_explanation", "")
    macro_text = macro_story.get("macro_text", "")
    heavyweight_text = macro_story.get("heavyweight_text", "")

    st.html(f"""
    <div class="monolith-card" style="border-left: 4px solid {regime_color}; padding:20px; margin-bottom:20px;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:14px; padding-bottom:12px; border-bottom:1px solid rgba(255,255,255,0.06);">
            <div style="display:flex; align-items:center; gap:12px;">
                <span style="font-size:24px;">🎙️</span>
                <div>
                    <span class="font-syncopate" style="font-size:13px; font-weight:700; color:#FFFFFF; letter-spacing:2px;">The Human Market Story</span>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:10px; color:#94A3B8; letter-spacing:1px;">FACTUAL REGIME ANALYSIS • INSTITUTIONAL DOSSIER</div>
                </div>
            </div>
            <span style="background:{regime_color}22; color:{regime_color}; border:1px solid {regime_color}55; font-size:11px; font-weight:800; padding:5px 14px; border-radius:20px; font-family:'JetBrains Mono', monospace;">
                ● {html.escape(regime_badge)}
            </span>
        </div>

        <div style="color:#F1F5F9; font-size:14.5px; line-height:1.65; margin-bottom:16px;">
            {headline}
        </div>

        <div class="macro-pillars-grid">
            <div class="macro-pillar-card" style="border-top:3px solid #10B981;">
                <div class="pillar-title" style="color:#10B981;">🎯 MATHEMATICAL REFERENCE LEVELS</div>
                <div class="pillar-body">{triggers}</div>
            </div>

            <div class="macro-pillar-card" style="border-top:3px solid #00F0FF;">
                <div class="pillar-title" style="color:#00F0FF;">⚡ INDIA VIX & VOLATILITY BAND</div>
                <div class="pillar-body">
                    {vix_status}<br>
                    {vix_explanation}
                </div>
            </div>

            <div class="macro-pillar-card" style="border-top:3px solid #F59E0B;">
                <div class="pillar-title" style="color:#F59E0B;">🛢️ CRUDE OIL & USD/INR MACRO COUPLING</div>
                <div class="pillar-body">{macro_text}</div>
            </div>

            <div class="macro-pillar-card" style="border-top:3px solid #FCD34D;">
                <div class="pillar-title" style="color:#FCD34D;">🏦 CONSTITUENT DRIVERS & BREADTH</div>
                <div class="pillar-body">{heavyweight_text}</div>
            </div>
        </div>
    </div>
    """)

    # 2. Master Hero Horizon Card
    prev_date = current.get("prev_date", "N/A")
    horizon_notice = f"THE APEX EXECUTION HORIZON (MATHEMATICALLY CALCULATED FROM PREVIOUS SESSION: {prev_date} OHLC)"
    if live_snap.get("is_post_close"):
        horizon_notice += " • NEXT-SESSION HORIZON ACTIVE"

    st.html(f"""
    <div class="monolith-card-gold" style="text-align:center;">
        <div style="display:flex; justify-content:center; gap:12px; margin-bottom:12px; flex-wrap:wrap;">
            <span title="Intraday strength score derived from 40% range position, 35% price change, and 25% constituent advance/decline breadth." style="font-family:'JetBrains Mono', monospace; font-size:11px; font-weight:800; color:#FCD34D; background:rgba(245,158,11,0.15); border:1px solid rgba(245,158,11,0.35); padding:4px 16px; border-radius:999px; cursor:help;">
                {html.escape(current.get('conf', 'Intraday Strength: Heuristic'))}
            </span>
            <span style="font-family:'JetBrains Mono', monospace; font-size:11px; font-weight:800; color:#6EE7B7; background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.35); padding:4px 16px; border-radius:999px;">
                {html.escape(current.get('date_info', ''))}
            </span>
        </div>

        <div class="font-syncopate" style="font-size: clamp(34px, 5.5vw, 64px); font-weight:700; margin:4px 0; color:#FFFFFF; line-height:1.1;">
            {html.escape(current.get('title', ''))} <span class="gold-gradient-text gold-glow">{html.escape(current.get('tag', ''))}</span>
        </div>

        <div style="display:flex; justify-content:center; align-items:center; gap:18px; margin:10px 0 16px 0; flex-wrap:wrap;">
            <span style="font-family:'JetBrains Mono', monospace; font-size: clamp(36px, 5.5vw, 62px); font-weight:900; color:#FFFFFF; letter-spacing:-2px;">
                {current.get('spot_str', 'N/A')}
            </span>
            <span style="font-family:'JetBrains Mono', monospace; font-size:14px; font-weight:800; color:#10B981; background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.3); padding:8px 16px; border-radius:12px;">
                {html.escape(current.get('chg', 'N/A'))}
            </span>
        </div>

        <div style="border-top: 1px solid rgba(255,255,255,0.06); padding-top:18px;">
            <div style="font-family:'Syncopate', sans-serif; font-size:10px; font-weight:700; color:#94A3B8; letter-spacing:3px; margin-bottom:14px;">
                {html.escape(horizon_notice)}
            </div>
            <div style="font-family:'JetBrains Mono', monospace; font-size:10px; color:#64748B;">
                ⚠️ STATUTORY NOTICE: Algorithmic levels (Reference S1, Pivot P, Reference R1, Reference R2) are mathematical reference points derived from prior session OHLC, strictly for quantitative research. Not investment advice or trading calls.
            </div>
        </div>
    </div>
    """)

    # 3. Four Neutral Quantitative Level Cards
    st.html(f"""
    <div class="runway-responsive-grid">
        <div class="monolith-card" style="border-top:3px solid #F43F5E; text-align:center; padding:16px 8px; margin-bottom:0;">
            <span style="font-family:'JetBrains Mono', monospace; font-size:10.5px; font-weight:800; color:#F43F5E; letter-spacing:1px;">📉 REFERENCE S1</span>
            <div style="font-family:'JetBrains Mono', monospace; font-size:clamp(19px, 2.5vw, 26px); font-weight:900; color:#FFFFFF; margin:4px 0;">{current.get('sl_str', 'N/A')}</div>
            <span style="font-size:10px; color:#94A3B8;">Prior Session Floor</span>
        </div>

        <div class="monolith-card" style="border-top:3px solid #F59E0B; text-align:center; padding:16px 8px; margin-bottom:0; box-shadow:0 0 25px rgba(245,158,11,0.15);">
            <span style="font-family:'JetBrains Mono', monospace; font-size:10.5px; font-weight:800; color:#F59E0B; letter-spacing:1px;">⚖️ PIVOT (P)</span>
            <div style="font-family:'JetBrains Mono', monospace; font-size:clamp(19px, 2.5vw, 26px); font-weight:900; color:#FFFFFF; margin:4px 0;">{current.get('piv_str', 'N/A')}</div>
            <span style="font-size:10px; color:#FCD34D;">Central Mean Level</span>
        </div>

        <div class="monolith-card" style="border-top:3px solid #10B981; text-align:center; padding:16px 8px; margin-bottom:0; box-shadow:0 0 30px rgba(16,185,129,0.18);">
            <span style="font-family:'JetBrains Mono', monospace; font-size:10.5px; font-weight:800; color:#10B981; letter-spacing:1px;">🎯 REFERENCE R1</span>
            <div style="font-family:'JetBrains Mono', monospace; font-size:clamp(19px, 2.5vw, 26px); font-weight:900; color:#FFFFFF; margin:4px 0;">{current.get('t1_str', 'N/A')}</div>
            <span style="font-family:'JetBrains Mono', monospace; font-size:11px; font-weight:800; color:#10B981;">{current.get('t1_diff', 'N/A')}</span>
        </div>

        <div class="monolith-card" style="border-top:3px solid #00F0FF; text-align:center; padding:16px 8px; margin-bottom:0; box-shadow:0 0 30px rgba(0,240,255,0.18);">
            <span style="font-family:'JetBrains Mono', monospace; font-size:10.5px; font-weight:800; color:#00F0FF; letter-spacing:1px;">🚀 REFERENCE R2</span>
            <div style="font-family:'JetBrains Mono', monospace; font-size:clamp(19px, 2.5vw, 26px); font-weight:900; color:#FFFFFF; margin:4px 0;">{current.get('t2_str', 'N/A')}</div>
            <span style="font-family:'JetBrains Mono', monospace; font-size:11px; font-weight:800; color:#00F0FF;">{current.get('t2_diff', 'N/A')}</span>
        </div>
    </div>
    """)

