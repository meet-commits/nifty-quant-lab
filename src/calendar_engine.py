"""
src/calendar_engine.py
Official NSE Trading Calendar & Expiry Management Engine.
Manages:
1. Master trading calendar (2019-2026) with NORMAL, HOLIDAY, and MUHURAT classifications.
2. Circular-sourced weekly & monthly expiry calendar per instrument.
3. Holiday precedence resolution (rolls back to preceding trading day when expiry is a holiday).
4. Exact calendar DTE calculation for intraday and overnight carry strategies.
"""

import os
from datetime import date, timedelta
from functools import lru_cache
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(BASE_DIR, "config")

CALENDAR_CSV = os.path.join(CONFIG_DIR, "nse_calendar.csv")
EXPIRY_CSV = os.path.join(CONFIG_DIR, "expiry_calendar.csv")


@lru_cache(maxsize=1)
def load_nse_calendar() -> pd.DataFrame:
    """Loads master NSE calendar with columns: date, type, name."""
    if os.path.exists(CALENDAR_CSV):
        df = pd.read_csv(CALENDAR_CSV)
        df['date_dt'] = pd.to_datetime(df['date']).dt.date
        return df
    return pd.DataFrame(columns=["date", "type", "name", "date_dt"])


@lru_cache(maxsize=1)
def load_expiry_calendar() -> pd.DataFrame:
    """Loads official expiry rules per instrument."""
    if os.path.exists(EXPIRY_CSV):
        return pd.read_csv(EXPIRY_CSV)
    return pd.DataFrame(columns=["instrument", "start_date", "end_date", "regular_expiry_weekday", "holiday_rule", "description"])


@lru_cache(maxsize=1)
def get_holiday_dates_frozenset() -> frozenset:
    """Returns frozen set of all dates where NSE trading was closed (HOLIDAY)."""
    cal = load_nse_calendar()
    if cal.empty:
        return frozenset()
    return frozenset(cal[cal["type"] == "HOLIDAY"]["date_dt"])


@lru_cache(maxsize=1)
def get_muhurat_dates_frozenset() -> frozenset:
    """Returns frozen set of all Muhurat trading session dates."""
    cal = load_nse_calendar()
    if cal.empty:
        return frozenset()
    return frozenset(cal[cal["type"] == "MUHURAT"]["date_dt"])


@lru_cache(maxsize=4096)
def get_contract_expiry_date(
    session_date: date,
    instrument: str = "NIFTY",
    for_overnight_carry: bool = False
) -> date:
    """
    Computes exact settlement expiry date for an option contract traded on session_date.
    - Resolves regular weekday from config/expiry_calendar.csv.
    - If scheduled expiry is an exchange holiday, rolls backward to preceding trading day.
    - If session_date is expiry day itself:
        * for_overnight_carry=True (S1 entry at 15:15): rolls to next week's contract.
        * for_overnight_carry=False (S2 intraday trade): expires today.
    """
    exp_df = load_expiry_calendar()
    s_str = session_date.strftime("%Y-%m-%d")
    matched = exp_df[(exp_df["instrument"] == instrument) & (exp_df["start_date"] <= s_str) & (exp_df["end_date"] >= s_str)]
    
    if matched.empty:
        reg_weekday = 3  # Thursday standard
    else:
        reg_weekday = int(matched.iloc[0]["regular_expiry_weekday"])

    holidays = get_holiday_dates_frozenset()

    def resolve_expiry(target_reg_date: date) -> date:
        cur = target_reg_date
        while cur in holidays or cur.weekday() >= 5:
            cur -= timedelta(days=1)
        return cur

    # Find the upcoming regular expiry weekday of the current week
    days_ahead = (reg_weekday - session_date.weekday()) % 7
    cur_week_reg = session_date + timedelta(days=days_ahead)
    cur_week_exp = resolve_expiry(cur_week_reg)

    if cur_week_exp < session_date:
        next_week_reg = cur_week_reg + timedelta(days=7)
        return resolve_expiry(next_week_reg)
    elif cur_week_exp == session_date:
        if for_overnight_carry:
            next_week_reg = cur_week_reg + timedelta(days=7)
            return resolve_expiry(next_week_reg)
        else:
            return cur_week_exp
    else:
        return cur_week_exp


def calculate_calendar_dte(
    session_date: date,
    instrument: str = "NIFTY",
    for_overnight_carry: bool = False
) -> int:
    """Returns calendar days from session_date to the contract expiry date."""
    exp_dt = get_contract_expiry_date(session_date, instrument, for_overnight_carry)
    return (exp_dt - session_date).days
