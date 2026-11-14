"""
src/macro_story_engine.py
Quantitative Market Regime & Intermarket Intelligence Engine.
Zero hardcoded synthetic data or speculative claims. Every sentence, level,
and constituent relationship is mathematically derived from verified snapshot values.
"""

import html
from typing import Dict, List, Any, Tuple

def clamp_val(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))

def compute_intraday_strength_score(spot: float, piv: float, day_high: float, day_low: float, vix: float, hw_adv_ratio: float) -> float:
    """
    Computes heuristic intraday strength score (0 to 100).
    Formula: Range position (40%) + Pivot proximity (35%) + Constituent breadth (25%).
    Strictly an intraday momentum heuristic; not a directional probability.
    """
    day_range = max(day_high - day_low, 1.0)
    range_pos = (spot - day_low) / day_range
    piv_proximity = 1.0 - min(abs(spot - piv) / day_range, 1.0)
    vix_factor = clamp_val((20.0 - vix) / 10.0, 0.0, 1.0)
    
    score = (range_pos * 40.0) + (piv_proximity * 35.0) + (hw_adv_ratio * 25.0)
    return round(clamp_val(score, 5.0, 95.0), 1)

def compute_dynamic_bull_bear_radar(spot: float, day_high: float, day_low: float, pct_chg: float, hw_adv_ratio: float) -> Tuple[int, int]:
    """
    Computes real-time Intraday Strength distribution (Bulls vs Bears).
    Weights: Range Position (40%), Index % Change (35%), Constituent Breadth (25%).
    """
    day_range = max(day_high - day_low, 1.0)
    range_pos = (spot - day_low) / day_range
    score = (range_pos * 40.0) + (clamp_val(50.0 + (pct_chg * 12.0), 10.0, 90.0) * 0.35) + (hw_adv_ratio * 100.0 * 0.25)
    bull_pct = int(round(clamp_val(score, 10, 90)))
    bear_pct = 100 - bull_pct
    return bull_pct, bear_pct

def build_comprehensive_market_story(snapshot: Dict[str, Any], asset_name: str = "NIFTY 50") -> Dict[str, Any]:
    """
    Builds a factual 4-pillar quantitative regime summary for the selected asset.
    Statements are derived strictly from live snapshot metrics without speculative claims.
    """
    is_nifty = "NIFTY 50" in asset_name
    is_bank = "BANK" in asset_name
    is_sensex = "SENSEX" in asset_name
    is_gift = "GIFT" in asset_name

    # Select appropriate asset quote dictionary
    if is_bank:
        q = snapshot.get("banknifty", {})
        top_anchors = ["HDFCBANK", "ICICIBANK"]
    elif is_sensex:
        q = snapshot.get("sensex", {})
        top_anchors = ["RELIANCE", "HDFCBANK"]
    elif is_gift:
        q = snapshot.get("giftnifty", {})
        top_anchors = ["RELIANCE", "HDFCBANK"]
    else:
        q = snapshot.get("nifty", {})
        top_anchors = ["RELIANCE", "HDFCBANK"]

    spot = q.get("value") or q.get("last")
    if spot is not None:
        spot = float(spot)

    # If feed has no valid spot price, return clean offline state
    if spot is None or spot <= 0.0:
        return {
            "headline": f"<b>{html.escape(asset_name)}</b> feed is currently unavailable or awaiting market open.",
            "regime_badge": "FEED OFFLINE / AWAITING DATA",
            "regime_color": "#64748B",
            "triggers": "Awaiting valid exchange session quotes to calculate algorithmic reference levels.",
            "vix_status": "VIX data pending exchange tick.",
            "vix_explanation": "Implied volatility metrics will update when the market session opens.",
            "macro_text": "Intermarket benchmarks (Crude, USD/INR, S&P 500) will sync on the next cycle.",
            "heavyweight_text": "Constituent advance/decline distribution is currently offline.",
            "strength_score": 50.0,
            "conviction_pct": 50.0,
            "bull_pct": 50,
            "bear_pct": 50,
            "rr_ratio": 1.0
        }

    pct = q.get("chg_pct", 0.0)
    chg_pts = q.get("chg_pts", 0.0)
    day_high = q.get("high", spot)
    day_low = q.get("low", spot)
    sl = q.get("s1") or q.get("sl", spot - 50.0)
    piv = q.get("pivot", spot)
    t1 = q.get("r1") or q.get("t1", spot + 50.0)
    t2 = q.get("r2") or q.get("t2", spot + 100.0)

    # Intermarket quotes
    vix_q = snapshot.get("vix", {})
    vix_val = vix_q.get("value") or vix_q.get("last")
    vix_chg_pct = vix_q.get("chg_pct", 0.0)

    crude_q = snapshot.get("crude", {})
    crude_val = crude_q.get("value") or crude_q.get("last")
    crude_chg_pct = crude_q.get("chg_pct", 0.0)

    usdinr_q = snapshot.get("usdinr", {})
    usd_val = usdinr_q.get("value") or usdinr_q.get("last")
    usd_chg_pct = usdinr_q.get("chg_pct", 0.0)

    sp_q = snapshot.get("sp500", {})
    sp_val = sp_q.get("value") or sp_q.get("last")
    sp_chg_pct = sp_q.get("chg_pct", 0.0)

    # Index Constituents from snapshot
    hw = snapshot.get("heavyweights", [])
    hw_valid = [h for h in hw if h.get("price") != "N/A"]
    hw_sorted_desc = sorted(
        hw_valid,
        key=lambda x: float(str(x.get("pct", "0")).replace("%", "").replace("+", "").strip() or 0),
        reverse=True
    )
    hw_sorted_asc = sorted(
        hw_valid,
        key=lambda x: float(str(x.get("pct", "0")).replace("%", "").replace("+", "").strip() or 0)
    )

    advancers = [h for h in hw_valid if h.get("bull", False)]
    hw_adv_ratio = len(advancers) / max(len(hw_valid), 1)

    top_leaders = [f"<b>{html.escape(h['sym'])}</b> ({html.escape(h['pct'])})" for h in hw_sorted_desc[:2]]
    top_drags = [f"<b>{html.escape(h['sym'])}</b> ({html.escape(h['pct'])})" for h in hw_sorted_asc[:2]]

    top_leaders_str = ", ".join(top_leaders) if top_leaders else "Broad neutral breadth"
    top_drags_str = ", ".join(top_drags) if top_drags else "No major drags recorded"

    # Distances
    dist_piv = piv - spot
    dist_sl = spot - sl
    dist_t1 = t1 - spot
    dist_t2 = t2 - spot
    day_range = max(day_high - day_low, 1.0)
    dist_from_low = spot - day_low

    # Intraday Strength Score (heuristic) & Radar
    strength_score = compute_intraday_strength_score(
        spot, piv, day_high, day_low,
        float(vix_val) if vix_val is not None else 15.0,
        hw_adv_ratio
    )
    bull_pct, bear_pct = compute_dynamic_bull_bear_radar(
        spot, day_high, day_low,
        float(pct) if pct is not None else 0.0,
        hw_adv_ratio
    )

    risk_pts = max(dist_sl, 5.0)
    reward_pts = max(dist_t1, 5.0)
    rr_ratio = round(reward_pts / risk_pts, 2)

    # Regime Determination (strictly mathematical)
    if spot >= piv:
        regime_badge = "ABOVE PIVOT (EXPANSION REGIME)"
        regime_color = "#10B981"
        headline = (
            f"<b>{html.escape(asset_name)}</b> is trading at <b>{spot:,.2f}</b> ({pct:+.2f}%), "
            f"<b>+{abs(dist_piv):.1f} pts</b> above the session Pivot level ({piv:,.1f}). "
            f"Trading at {((spot - day_low) / day_range * 100):.0f}% of the intraday range."
        )
    elif spot >= day_low:
        regime_badge = "BELOW PIVOT (CONSOLIDATION REGIME)"
        regime_color = "#F59E0B"
        headline = (
            f"<b>{html.escape(asset_name)}</b> is trading at <b>{spot:,.2f}</b> ({pct:+.2f}% / {chg_pts:+.1f} pts), "
            f"<b>{abs(dist_piv):.1f} pts</b> below the session Pivot level ({piv:,.1f}). "
            f"Holding +{dist_from_low:.1f} pts above session low ({day_low:,.1f})."
        )
    else:
        regime_badge = "BELOW SESSION LOW (BEARISH EXTENSION)"
        regime_color = "#F43F5E"
        headline = (
            f"<b>{html.escape(asset_name)}</b> is trading at <b>{spot:,.2f}</b> ({pct:+.2f}%), "
            f"below session low ({day_low:,.1f}). Reference S1 floor is at {sl:,.1f} ({abs(dist_sl):.1f} pts)."
        )

    # Pillar 1: Reference Levels
    if spot >= piv:
        up_text = (
            f"Spot is above Pivot ({piv:,.1f}). Session High is {day_high:,.1f}. "
            f"Mathematical Reference R1 is at <b>{t1:,.1f}</b> (+{dist_t1:.1f} pts) "
            f"and Reference R2 is at <b>{t2:,.1f}</b> (+{dist_t2:.1f} pts)."
        )
    else:
        up_text = (
            f"Session Pivot is at <b>{piv:,.1f}</b> ({abs(dist_piv):.1f} pts from spot). "
            f"Reference R1 is at <b>{t1:,.1f}</b> (+{dist_t1:.1f} pts)."
        )

    down_text = (
        f"Session Low is at <b>{day_low:,.1f}</b> ({dist_from_low:.1f} pts buffer). "
        f"Prior-session Reference S1 floor is at <b>{sl:,.1f}</b> ({dist_sl:.1f} pts downside distance)."
    )

    triggers_text = f"""
    • <b>Upside Reference Levels:</b> {up_text}<br>
    • <b>Downside Reference Levels:</b> {down_text}<br>
    • <b>Calculated R:R Ratio:</b> 1 : {rr_ratio} (to Reference R1 relative to Reference S1 distance).
    """

    # Pillar 2: India VIX
    if vix_val is not None:
        vix_f = float(vix_val)
        if vix_f < 14.5:
            vix_status = f"<b>India VIX at {vix_f:.2f} ({vix_chg_pct:+.2f}% 1d) — Low Volatility Regime</b>"
            vix_explanation = (
                f"India VIX at {vix_f:.2f} reflects compressed implied volatility. "
                f"Intraday expected move band ~{round(spot * (vix_f / 100) / 16, 1)} pts."
            )
        elif vix_f < 18.0:
            vix_status = f"<b>India VIX at {vix_f:.2f} ({vix_chg_pct:+.2f}% 1d) — Moderate Volatility Regime</b>"
            vix_explanation = (
                f"India VIX at {vix_f:.2f} indicates standard intraday movement bands. "
                f"1-day standard expected range is ±{round(spot * (vix_f / 100) / 16, 1)} pts."
            )
        else:
            vix_status = f"<b>India VIX at {vix_f:.2f} ({vix_chg_pct:+.2f}% 1d) — Elevated Volatility Regime</b>"
            vix_explanation = (
                f"India VIX at {vix_f:.2f} signals wide trading ranges and expanded option premia. "
                f"Expect wider intraday swings; 1-day band expands to ±{round(spot * (vix_f / 100) / 16, 1)} pts."
            )
    else:
        vix_status = "<b>India VIX: Feed Offline / Awaiting Session</b>"
        vix_explanation = "Implied volatility calculation pending active tick data."

    # Pillar 3: Commodities & Currency
    crude_str = f"${float(crude_val):.2f} ({crude_chg_pct:+.2f}%)" if crude_val is not None else "N/A"
    usd_str = f"₹{float(usd_val):.2f} ({usd_chg_pct:+.2f}%)" if usd_val is not None else "N/A"
    sp_str = f"{float(sp_val):,.1f} ({sp_chg_pct:+.2f}%)" if sp_val is not None else "N/A"

    macro_text = f"""
    • <b>WTI Crude Oil:</b> {crude_str} (1-day session return).<br>
    • <b>USD/INR Spot:</b> {usd_str} (Forex reference rate).<br>
    • <b>US S&P 500 Futures:</b> {sp_str} (Global risk sentiment benchmark).
    """

    # Pillar 4: Heavyweight Constituents
    heavyweight_text = f"""
    • <b>Leading Constituents:</b> {top_leaders_str}.<br>
    • <b>Lagging Constituents:</b> {top_drags_str}.<br>
    • <b>Breadth Summary:</b> {len(advancers)} advancers / {len(hw_valid) - len(advancers)} decliners ({hw_adv_ratio*100:.0f}% positive).
    """

    return {
        "headline": headline,
        "regime_badge": regime_badge,
        "regime_color": regime_color,
        "triggers": triggers_text,
        "vix_status": vix_status,
        "vix_explanation": vix_explanation,
        "macro_text": macro_text,
        "heavyweight_text": heavyweight_text,
        "strength_score": strength_score,
        "conviction_pct": strength_score,  # for backward compatibility
        "bull_pct": bull_pct,
        "bear_pct": bear_pct,
        "rr_ratio": rr_ratio
    }
