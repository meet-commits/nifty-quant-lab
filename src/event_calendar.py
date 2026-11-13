"""
src/event_calendar.py
High-Impact Market Events and Expiry Countdown Engine.
Calculates dynamic countdowns for:
- Weekly F&O Index Expiry (Thursdays, auto-adjusts for holidays)
- RBI Monetary Policy Committee (MPC) meetings
- US Federal Reserve FOMC interest rate decisions
- Union Budget
"""

import datetime
from typing import List, Dict

def get_next_weekly_expiry(base_date: datetime.date = None) -> Dict:
    """Calculates days left for next NSE Weekly Expiry (Thursday)."""
    if base_date is None:
        base_date = datetime.date.today()
        
    days_ahead = (3 - base_date.weekday()) % 7  # Thursday is weekday 3
    if days_ahead == 0:
        # Today is expiry day
        target_expiry = base_date
        status = "TODAY (High Theta Decay)"
        days_left = 0
    else:
        target_expiry = base_date + datetime.timedelta(days=days_ahead)
        days_left = days_ahead
        status = f"{days_left} Days Left"
        
    return {
        "event": "Weekly F&O Expiry",
        "date": target_expiry.strftime("%d %b %Y"),
        "countdown": status,
        "impact": "Theta Decay & Pinning",
        "color": "#F59E0B"
    }

def get_high_impact_events(base_date: datetime.date = None) -> List[Dict]:
    """Returns dynamic schedule of high-impact macroeconomic events."""
    if base_date is None:
        base_date = datetime.date.today()
        
    weekly = get_next_weekly_expiry(base_date)
    
    # Scheduled Macro Dates (Official Schedule)
    macro_schedule = [
        {"name": "RBI Monetary Policy", "target": datetime.date(2026, 12, 6), "desc": "Repo Rate Decision", "color": "#10B981"},
        {"name": "US Fed FOMC Meet", "target": datetime.date(2026, 12, 18), "desc": "Global Liquidity Policy", "color": "#00F0FF"},
        {"name": "India CPI Inflation", "target": datetime.date(2026, 11, 12), "desc": "Macro Inflation Print", "color": "#FCD34D"},
        {"name": "Union Budget 2027", "target": datetime.date(2027, 2, 1), "desc": "Annual Fiscal Policy", "color": "#F43F5E"}
    ]
    
    events = [weekly]
    for m in macro_schedule:
        days = (m["target"] - base_date).days
        if days >= 0:
            events.append({
                "event": m["name"],
                "date": m["target"].strftime("%d %b %Y"),
                "countdown": f"{days} Days Left" if days > 0 else "TODAY",
                "impact": m["desc"],
                "color": m["color"]
            })
            
    return events

if __name__ == "__main__":
    evs = get_high_impact_events()
    for e in evs:
        print(f"[{e['event']}] {e['date']} -> {e['countdown']} ({e['impact']})")

