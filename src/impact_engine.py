"""
src/impact_engine.py
News Authenticity, Credibility Verification & Market Impact Scoring Engine.
Filters retail noise, verifies source tier, and scores institutional market impact (1 to 10).
"""

import re
from typing import Dict

# Trusted Tier 1 Official & Financial Wires
TRUSTED_SOURCES = [
    "reuters", "bloomberg", "pti", "press trust of india", "economic times",
    "the economic times", "livemint", "mint", "business standard",
    "ndtv profit", "financial express", "moneycontrol", "cnbc-tv18", "rbi", "sebi", "nse"
]

# High-Impact Macro & Index Catalyst Keywords
HIGH_IMPACT_RULES = [
    {"pattern": r"\b(rbi|repo rate|monetary policy|mpc)\b", "weight": 9.5, "cat": "Central Bank Policy", "move_pts": "200 - 450 pts", "verdict": "CRITICAL MACRO DRIVER"},
    {"pattern": r"\b(fed|fomc|jerome powell|interest rate cut|rate hike)\b", "weight": 9.0, "cat": "Global Liquidity", "move_pts": "180 - 350 pts", "verdict": "GLOBAL RISK-ON/OFF"},
    {"pattern": r"\b(crude|oil prices|brent|opec)\b", "weight": 8.5, "cat": "Macro Commodity", "move_pts": "150 - 300 pts", "verdict": "RUPEE & INFLATION MOVER"},
    {"pattern": r"\b(union budget|budget 2027|stt|capital gains tax)\b", "weight": 10.0, "cat": "Fiscal Policy", "move_pts": "300 - 600 pts", "verdict": "MEGA STRUCTURAL EVENT"},
    {"pattern": r"\b(war|geopolitical|missile|sanction|ceasefire)\b", "weight": 9.5, "cat": "Geopolitical Shock", "move_pts": "250 - 500 pts", "verdict": "HIGH VOLATILITY SPIKE"},
    {"pattern": r"\b(hdfc bank|reliance|icici bank|tcs|infosys)\b.*(q[1-4]|earnings|results|profit|loss)", "weight": 8.5, "cat": "Heavyweight Earnings", "move_pts": "120 - 250 pts", "verdict": "DIRECT INDEX WEIGHT MOVER"},
    {"pattern": r"\b(fii|foreign institutional|fpi)\b.*(dump|sell|buy|outflow|inflow)", "weight": 8.0, "cat": "Institutional Flow", "move_pts": "100 - 200 pts", "verdict": "LIQUIDITY EXPANSION/DRAIN"}
]

# Low Impact / Noise Keywords
NOISE_PATTERNS = [
    r"\b(target price|brokerage gives|analyst recommends|multibagger|penny stock|whatsapp|shares to buy today|expert advice)\b"
]

def verify_and_score_news(title: str, source: str = "") -> Dict:
    """
    Evaluates news credibility, source tier, and quant impact score (1 to 10).
    """
    title_lower = title.lower()
    source_lower = source.lower()

    # 1. Source Credibility Check
    is_trusted_source = any(ts in source_lower for ts in TRUSTED_SOURCES)
    credibility_badge = "TIER 1 (OFFICIAL / VERIFIED)" if is_trusted_source else "TIER 2 (MEDIA REPORT)"

    # Check for retail noise
    for np in NOISE_PATTERNS:
        if re.search(np, title_lower):
            return {
                "title": title,
                "source": source,
                "credibility": "RETAIL NOISE / OPINION",
                "impact_score": 2.0,
                "impact_level": "LOW NOISE",
                "expected_move": "< 30 pts (Ignore)",
                "category": "Retail Chatter",
                "action": "IGNORE: Retail traps & broker opinions don't move Nifty.",
                "color": "#64748B"
            }

    # 2. Match against High-Impact Rules
    max_weight = 3.5
    matched_cat = "General Market Flow"
    move_est = "40 - 80 pts"
    action_note = "Standard market fluctuation; wait for Pivot floor test."
    badge_color = "#00F0FF"

    for rule in HIGH_IMPACT_RULES:
        if re.search(rule["pattern"], title_lower):
            if rule["weight"] > max_weight:
                max_weight = rule["weight"]
                matched_cat = rule["cat"]
                move_est = rule["move_pts"]
                action_note = rule["verdict"]

    if max_weight >= 8.5:
        impact_level = "HIGH IMPACT (INDEX MOVER)"
        badge_color = "#F43F5E"
    elif max_weight >= 6.5:
        impact_level = "MEDIUM IMPACT (SECTOR MOVER)"
        badge_color = "#F59E0B"
    else:
        impact_level = "LOW NOISE"
        badge_color = "#10B981"

    return {
        "title": title,
        "source": source if source else "Financial Wire",
        "credibility": credibility_badge,
        "impact_score": max_weight,
        "impact_level": impact_level,
        "expected_move": move_est,
        "category": matched_cat,
        "action": action_note,
        "color": badge_color
    }

if __name__ == "__main__":
    test_headlines = [
        ("RBI MPC keeps repo rate unchanged at 6.5%, maintains withdrawal of accommodation", "Reuters"),
        ("Crude Oil surges 4.2% overnight as Middle East tensions escalate", "Bloomberg"),
        ("HDFC Bank Q2 Net Profit jumps 16% YoY to ₹16,820 Crore", "Economic Times"),
        ("Top 5 penny stocks to buy today for 100% multibagger gains", "Random Blog"),
        ("Stock market holiday today for Diwali Balipratipada", "NDTV")
    ]
    for h, s in test_headlines:
        res = verify_and_score_news(h, s)
        print(f"[{res['impact_level']} | Score: {res['impact_score']}/10] {res['title']} -> {res['expected_move']} ({res['action']})")

