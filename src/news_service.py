"""
src/news_service.py
Real-Time Market News Ingestion & Institutional Impact Scoring Engine.
Pulls genuine live headlines from Google News India Financial RSS.
Uses defusedxml for security, st.cache_data for 300s caching, and escapes all links/strings.
Zero hardcoded/fake fallback news: returns empty list on network failure.
"""

import os
import sys
import html
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict

try:
    import defusedxml.ElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

try:
    import streamlit as st
    cache_decorator = st.cache_data(ttl=300)
except Exception:
    # Fallback decorator if running outside Streamlit runtime
    def cache_decorator(func):
        return func

SRC_PATH = os.path.dirname(os.path.abspath(__file__))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from impact_engine import verify_and_score_news

IST = ZoneInfo("Asia/Kolkata")


@cache_decorator
def fetch_live_market_news(limit: int = 5) -> List[Dict]:
    """
    Fetches real-time financial headlines and evaluates quantitative impact.
    Returns empty list [] on network or parsing failure (never fake seed headlines).
    """
    url = "https://news.google.com/rss/search?q=Nifty+Sensex+Stock+Market+India&hl=en-IN&gl=IN&ceid=IN:en"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "NiftyQuantLab/2.0 (Mozilla/5.0; Windows NT 10.0)"}
    )

    news_list = []
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        items = root.findall(".//item")[:limit]

        for item in items:
            raw_title = item.find("title").text if item.find("title") is not None else ""
            raw_pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
            raw_link = item.find("link").text if item.find("link") is not None else "#"

            clean_title = raw_title.strip()
            source = "Financial Wire"
            if " - " in clean_title:
                parts = clean_title.rsplit(" - ", 1)
                clean_title = parts[0].strip()
                source = parts[1].strip()

            # Format publication time
            pub_time_display = ""
            if raw_pub_date:
                try:
                    # RFC 822 format: e.g. "Sat, 14 Nov 2026 05:30:00 GMT"
                    dt = datetime.strptime(raw_pub_date[:25].strip(), "%a, %d %b %Y %H:%M:%S")
                    dt_ist = dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(IST)
                    pub_time_display = dt_ist.strftime("%d %b, %I:%M %p IST")
                except Exception:
                    pub_time_display = raw_pub_date[:16]
            if not pub_time_display:
                pub_time_display = datetime.now(IST).strftime("%d %b, %I:%M %p IST")

            # Run Institutional Impact & Authenticity Evaluation
            eval_res = verify_and_score_news(clean_title, source)

            news_list.append({
                "title": html.escape(clean_title),
                "source": html.escape(source),
                "time": html.escape(pub_time_display),
                "tag": html.escape(eval_res["impact_level"]),
                "color": eval_res["color"],
                "score": eval_res["impact_score"],
                "credibility": html.escape(eval_res["credibility"]),
                "expected_move": html.escape(eval_res["expected_move"]),
                "action": html.escape(eval_res["action"]),
                "link": html.escape(raw_link.strip())
            })
    except Exception as e:
        print(f"⚠️ Live market news fetch error: {e}")
        # Always return empty list on failure — NEVER fabricated/fake news items
        return []

    return news_list


if __name__ == "__main__":
    items = fetch_live_market_news()
    print(f"Fetched {len(items)} live articles:")
    for it in items:
        print(f"[{it['tag']} | Score: {it['score']}/10] {it['source']} ({it['time']}): {it['title']}")
        print(f"   Link: {it['link']}")
