"""
src/live_market_service.py
High-performance quantitative live market engine for Indian Indices, Heavyweights, and Scanners.
Integrates:
  - Official NSE MCP Server as PRIMARY equity feed (mcp.nseindia.in)
  - Batched Yahoo Finance feeds with exponential backoff & in-memory caching
  - Standardized Quote dataclass (src/datapoint.py) with strict stale detection (>300s IST)
  - Asia/Kolkata timezone and config/nse_holidays.csv exchange calendar
  - Mathematically fixed floor pivots computed from PREVIOUS completed session (DuckDB)
  - Zero hardcoded fallback prices: returns None / 'N/A' on feed failure
  - Zero synthetic random candle generation
"""

import os
import time
import concurrent.futures
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import duckdb
import requests
import yfinance as yf

try:
    from datapoint import Quote, make_quote
    from nse_mcp_client import check_nse_mcp_health, fetch_nse_mcp_stock_quote, get_nse_mcp_stats
except ImportError:
    from src.datapoint import Quote, make_quote
    from src.nse_mcp_client import check_nse_mcp_health, fetch_nse_mcp_stock_quote, get_nse_mcp_stats

IST = ZoneInfo("Asia/Kolkata")
MARKET_CLOSE_TIME = dtime(15, 30)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
DB_PATH = os.path.join(DATA_DIR, "research.duckdb")
HOLIDAYS_PATH = os.path.join(CONFIG_DIR, "nse_holidays.csv")
CALENDAR_PATH = os.path.join(CONFIG_DIR, "nse_calendar.csv")
WEIGHTS_PATH = os.path.join(CONFIG_DIR, "weights.csv")

def load_index_weights_map() -> Dict[str, List[Dict[str, Any]]]:
    """Loads index constituent weights from config/weights.csv."""
    if not os.path.exists(WEIGHTS_PATH):
        return {"NIFTY 50": HEAVYWEIGHT_META}

    import csv
    weights_map: Dict[str, List[Dict[str, Any]]] = {}
    with open(WEIGHTS_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = row["index_symbol"].strip()
            if idx not in weights_map:
                weights_map[idx] = []
            weights_map[idx].append({
                "sym": row["symbol"].strip(),
                "yf": row["yf_ticker"].strip(),
                "name": row["name"].strip(),
                "wt": f"{float(row['weight_pct']):.1f}%",
                "weight_pct": float(row["weight_pct"]),
                "sector": row.get("sector", "").strip(),
                "as_of_date": row.get("as_of_date", "").strip()
            })
    return weights_map

# Global Cache Storage
_CACHE = {
    "last_fetch_time": 0.0,
    "snapshot": None,
    "candles_5m": {}
}
CACHE_TTL_SECONDS = 30.0  # 30-second in-memory throttle cache

_NSE_SESSION = None
_GIFT_CACHE = {
    "last_time": 0.0,
    "data": None
}

SYMBOL_TO_DB_MAP = {
    "^NSEI": "NIFTY_SPOT",
    "^NSEBANK": "BANKNIFTY_SPOT",
    "^BSESN": "SENSEX_SPOT",
    "^INDIAVIX": "INDIA_VIX",
    "CL=F": "CRUDE_OIL",
    "USDINR=X": "USD_INR",
    "ES=F": "SP500_FUT"
}

INDEX_GLOBAL_TICKERS = [
    "^NSEI", "^NSEBANK", "^BSESN", "^INDIAVIX", "CL=F", "USDINR=X", "ES=F"
]

HEAVYWEIGHT_META = [
    {"sym": "HDFCBANK", "yf": "HDFCBANK.NS", "wt": "11.2%", "name": "HDFC Bank Ltd"},
    {"sym": "RELIANCE", "yf": "RELIANCE.NS", "wt": "9.8%", "name": "Reliance Industries"},
    {"sym": "ICICIBANK", "yf": "ICICIBANK.NS", "wt": "7.9%", "name": "ICICI Bank Ltd"},
    {"sym": "SBIN", "yf": "SBIN.NS", "wt": "3.1%", "name": "State Bank of India"},
    {"sym": "LT", "yf": "LT.NS", "wt": "3.5%", "name": "Larsen & Toubro Ltd"},
    {"sym": "AXISBANK", "yf": "AXISBANK.NS", "wt": "3.8%", "name": "Axis Bank Ltd"},
    {"sym": "TCS", "yf": "TCS.NS", "wt": "4.2%", "name": "Tata Consultancy Services"},
    {"sym": "INFY", "yf": "INFY.NS", "wt": "5.6%", "name": "Infosys Ltd"}
]

CHARTINK_STOCK_MAP = {
    "SBIN": "SBIN.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "AXISBANK": "AXISBANK.NS",
    "TRENT": "TRENT.NS",
    "BEL": "BEL.NS",
    "DIXON": "DIXON.NS",
    "INDUSINDBK": "INDUSINDBK.NS",
    "TATACONSUM": "TATACONSUM.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "LT": "LT.NS",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "WIPRO": "WIPRO.NS"
}


def load_nse_holidays_map() -> Dict[str, str]:
    """Loads NSE holiday dates and names from config/nse_calendar.csv or config/nse_holidays.csv."""
    holidays = {}
    if os.path.exists(CALENDAR_PATH):
        try:
            df = pd.read_csv(CALENDAR_PATH)
            for _, r in df[df["type"] == "HOLIDAY"].iterrows():
                h_name = str(r["name"]).strip()
                if h_name != "Weekend Closure":
                    holidays[str(r["date"]).strip()] = h_name
            if holidays:
                return holidays
        except Exception:
            pass
    if os.path.exists(HOLIDAYS_PATH):
        try:
            df = pd.read_csv(HOLIDAYS_PATH)
            for _, r in df.iterrows():
                d_str = str(r["date"]).strip()
                h_name = str(r["holiday_name"]).strip()
                holidays[d_str] = h_name
        except Exception:
            pass
    return holidays


def get_market_session_status() -> Tuple[bool, str, str]:
    """
    Determines if Indian Equity Market (NSE/BSE) is currently in live trading hours.
    Evaluated strictly in Asia/Kolkata timezone.
    Cross-checks against NSE Holiday Calendar.
    """
    now = datetime.now(IST)
    weekday = now.weekday()  # 0=Monday, 4=Friday, 5=Saturday, 6=Sunday
    current_time = now.time()
    today_str = now.strftime("%Y-%m-%d")

    # Holiday Calendar check
    holidays = load_nse_holidays_map()
    if today_str in holidays:
        h_name = holidays[today_str]
        return False, f"EXCHANGE HOLIDAY: {h_name.upper()}", "#64748B"

    if weekday in range(0, 5):
        open_time = dtime(9, 15)
        close_time = dtime(15, 30)
        pre_open = dtime(9, 0)

        if open_time <= current_time <= close_time:
            return True, "LIVE TRADING OPEN (NSE/BSE)", "#10B981"
        elif pre_open <= current_time < open_time:
            return False, "PRE-MARKET DISCOVERY (OPENS 09:15 IST)", "#F59E0B"
        elif current_time > close_time:
            return False, "POST-MARKET (SESSION CLOSED)", "#94A3B8"
        else:
            return False, "OVERNIGHT QUEUE (OPENS 09:15 IST)", "#64748B"
    else:
        day_name = now.strftime("%A")
        return False, f"WEEKEND CLOSURE ({day_name.upper()})", "#64748B"


def get_last_persisted_quote(db_sym: str) -> Optional[Quote]:
    """Retrieves the last known completed session price from DuckDB with its timestamp."""
    if not os.path.exists(DB_PATH):
        return None
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        row = con.execute("""
            SELECT close, open, high, low, volume, date 
            FROM daily_prices 
            WHERE symbol = ? 
            ORDER BY date DESC 
            LIMIT 1;
        """, [db_sym]).fetchone()
        con.close()
        if row and row[0] is not None:
            c, o, h, l, v, d = row
            session_dt = datetime.combine(d, dtime(15, 30), tzinfo=IST)
            return make_quote(
                value=float(c),
                ts_ist=session_dt,
                source="DUCKDB_PERSISTED_EOD",
                is_market_hours=True,
                open=float(o),
                high=float(h),
                low=float(l),
                prev=float(c),
                volume=int(v) if v else 0,
                is_real=True,
                metadata={"persisted_date": str(d)}
            )
    except Exception:
        pass
    return None


def get_previous_session_ohlc(symbol: str = "NIFTY_SPOT") -> Optional[Dict[str, Any]]:
    """
    Queries the last completed trading session's High, Low, Close from DuckDB.
    Guarantees mathematically fixed floor pivots during market hours.
    """
    if not os.path.exists(DB_PATH):
        return None
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        row = con.execute("""
            SELECT high, low, close, date 
            FROM daily_prices 
            WHERE symbol = ? 
            ORDER BY date DESC 
            LIMIT 1;
        """, [symbol]).fetchone()
        con.close()
        if row and row[0] is not None and row[1] is not None and row[2] is not None:
            return {
                "high": float(row[0]),
                "low": float(row[1]),
                "close": float(row[2]),
                "date": str(row[3])
            }
    except Exception:
        pass
    return None


def calc_floor_pivots(high: Optional[float], low: Optional[float], close: Optional[float]) -> Dict[str, Optional[float]]:
    """
    Computes classical floor pivot levels:
    Pivot = (H + L + C) / 3
    R1 = 2*P - L, S1 = 2*P - H
    R2 = P + (H - L), S2 = P - (H - L)
    """
    if high is None or low is None or close is None or high <= 0 or low <= 0 or close <= 0:
        return {
            "pivot": None, "s1": None, "sl": None, "s2": None,
            "r1": None, "t1": None, "r2": None, "t2": None
        }
    p = (high + low + close) / 3.0
    r1 = 2.0 * p - low
    r2 = p + (high - low)
    s1 = 2.0 * p - high
    s2 = p - (high - low)
    return {
        "pivot": round(p, 1),
        "s1": round(s1, 1),
        "sl": round(s1, 1),
        "s2": round(s2, 1),
        "r1": round(r1, 1),
        "t1": round(r1, 1),
        "r2": round(r2, 1),
        "t2": round(r2, 1)
    }


def fetch_nse_gift_nifty(spot_nifty: Optional[float] = None) -> Dict[str, Any]:
    """
    Fetches verified real-time GIFT NIFTY contract quote directly from official NSE Market Status feed.
    Never invents artificial High/Low or basis offsets. If unavailable, sets unknown fields to None.
    """
    global _NSE_SESSION, _GIFT_CACHE
    now_ts = time.time()
    if _GIFT_CACHE["data"] is not None and (now_ts - _GIFT_CACHE["last_time"] < 25.0):
        return _GIFT_CACHE["data"]

    try:
        if _NSE_SESSION is None:
            _NSE_SESSION = requests.Session()
            _NSE_SESSION.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Referer': 'https://www.nseindia.com/',
                'Accept-Language': 'en-US,en;q=0.9',
            })
            try:
                _NSE_SESSION.get('https://www.nseindia.com/', timeout=3)
            except Exception:
                pass

        res = _NSE_SESSION.get('https://www.nseindia.com/api/marketStatus', timeout=4)
        if res.status_code == 200:
            payload = res.json()
            gift_obj = payload.get("giftnifty", {})
            if gift_obj and "LASTPRICE" in gift_obj:
                raw_last = gift_obj.get("LASTPRICE")
                last_p = float(raw_last) if raw_last is not None else None
                raw_chg = gift_obj.get("DAYCHANGE")
                day_chg = float(raw_chg) if raw_chg is not None else None
                raw_pct = gift_obj.get("PERCHANGE")
                pct_chg = float(raw_pct) if raw_pct is not None else None
                timestamp_str = gift_obj.get("TIMESTMP", datetime.now(IST).strftime("%d-%b-%Y %H:%M"))
                expiry = gift_obj.get("EXPIRYDATE", "")
                contracts = gift_obj.get("CONTRACTSTRADED", 0)

                high_p = None
                low_p = None
                open_p = None

                # Attempt to retrieve verified derivative prices
                try:
                    r_der = _NSE_SESSION.get('https://www.nseindia.com/api/liveEquity-derivatives?index=nse50_fut', timeout=3)
                    if r_der.status_code == 200:
                        der_data = r_der.json().get('data', [])
                        if der_data and len(der_data) > 0:
                            f1 = der_data[0]
                            high_p = float(f1.get('highPrice')) if f1.get('highPrice') else None
                            low_p = float(f1.get('lowPrice')) if f1.get('lowPrice') else None
                            open_p = float(f1.get('openPrice')) if f1.get('openPrice') else None
                except Exception:
                    pass

                result = {
                    "last": round(last_p, 2) if last_p is not None else None,
                    "chg_pts": round(day_chg, 2) if day_chg is not None else None,
                    "chg_pct": round(pct_chg, 2) if pct_chg is not None else None,
                    "chg_str": f"{'+' if day_chg >= 0 else ''}{day_chg:.2f}" if day_chg is not None else "N/A",
                    "pct_str": f"{'+' if pct_chg >= 0 else ''}{pct_chg:.2f}%" if pct_chg is not None else "N/A",
                    "high": round(high_p, 2) if high_p is not None else None,
                    "low": round(low_p, 2) if low_p is not None else None,
                    "open": round(open_p, 2) if open_p is not None else None,
                    "prev": round(last_p - day_chg, 2) if (last_p is not None and day_chg is not None) else None,
                    "bull": (day_chg >= 0) if day_chg is not None else True,
                    "timestamp": timestamp_str,
                    "expiry": expiry,
                    "contracts": contracts,
                    "is_live": True,
                    "is_real": True,
                    "source": "OFFICIAL_NSE_IX_FEED"
                }
                _GIFT_CACHE["data"] = result
                _GIFT_CACHE["last_time"] = now_ts
                return result
    except Exception:
        _NSE_SESSION = None

    # Honest failure return: never invent +28.5 or fake numbers
    return {
        "last": None,
        "chg_pts": None,
        "chg_pct": None,
        "chg_str": "N/A",
        "pct_str": "N/A",
        "high": None,
        "low": None,
        "open": None,
        "prev": None,
        "bull": True,
        "timestamp": datetime.now(IST).strftime("%d-%b-%Y %H:%M"),
        "expiry": "",
        "contracts": 0,
        "is_live": False,
        "is_real": False,
        "source": "UNAVAILABLE"
    }


def fetch_batch_yfinance_quotes(symbols: List[str], max_retries: int = 2) -> Dict[str, Quote]:
    """
    Downloads index and global tickers in batch via yf.download with timeout and caching.
    Falls back to DuckDB persisted data on connection failure.
    """
    quotes: Dict[str, Quote] = {}
    now_ist = datetime.now(IST)
    is_mkt_open, _, _ = get_market_session_status()

    for attempt in range(max_retries):
        try:
            df = yf.download(" ".join(symbols), period="5d", progress=False, timeout=8)
            if not df.empty:
                for sym in symbols:
                    try:
                        if isinstance(df.columns, pd.MultiIndex):
                            if 'Close' in df.columns.levels[0] and sym in df['Close'].columns:
                                c_s = df['Close'][sym].dropna()
                                if not c_s.empty:
                                    lp = float(c_s.iloc[-1])
                                    prev = float(c_s.iloc[-2]) if len(c_s) > 1 else lp
                                    hp = float(df['High'][sym].dropna().iloc[-1]) if ('High' in df.columns.levels[0] and sym in df['High'].columns and not df['High'][sym].dropna().empty) else lp
                                    lowp = float(df['Low'][sym].dropna().iloc[-1]) if ('Low' in df.columns.levels[0] and sym in df['Low'].columns and not df['Low'][sym].dropna().empty) else lp
                                    op = float(df['Open'][sym].dropna().iloc[-1]) if ('Open' in df.columns.levels[0] and sym in df['Open'].columns and not df['Open'][sym].dropna().empty) else lp

                                    quotes[sym] = make_quote(
                                        value=lp,
                                        ts_ist=now_ist,
                                        source="LIVE_YFINANCE_BATCH",
                                        is_market_hours=is_mkt_open,
                                        open=op,
                                        high=hp,
                                        low=lowp,
                                        prev=prev,
                                        is_real=True
                                    )
                        else:
                            c_s = df['Close'].dropna()
                            if not c_s.empty:
                                lp = float(c_s.iloc[-1])
                                prev = float(c_s.iloc[-2]) if len(c_s) > 1 else lp
                                quotes[sym] = make_quote(
                                    value=lp,
                                    ts_ist=now_ist,
                                    source="LIVE_YFINANCE_BATCH",
                                    is_market_hours=is_mkt_open,
                                    open=float(df['Open'].dropna().iloc[-1]) if not df['Open'].dropna().empty else lp,
                                    high=float(df['High'].dropna().iloc[-1]) if not df['High'].dropna().empty else lp,
                                    low=float(df['Low'].dropna().iloc[-1]) if not df['Low'].dropna().empty else lp,
                                    prev=prev,
                                    is_real=True
                                )
                    except Exception:
                        pass
                if len(quotes) >= len(symbols) * 0.7:
                    break
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1.0 * (2 ** attempt))

    # For any symbol that failed, query last persisted quote from DuckDB or set None
    for sym in symbols:
        if sym not in quotes:
            db_sym = SYMBOL_TO_DB_MAP.get(sym)
            persisted = get_last_persisted_quote(db_sym) if db_sym else None
            if persisted:
                quotes[sym] = persisted
            else:
                quotes[sym] = make_quote(
                    value=None,
                    ts_ist=now_ist,
                    source="UNAVAILABLE",
                    is_market_hours=is_mkt_open,
                    is_real=False
                )

    return quotes


def fetch_equity_quote(sym_code: str, yf_sym: str, is_mkt_open: bool) -> Tuple[str, Quote]:
    """
    Fetches equity quote with NSE MCP as PRIMARY source, yfinance as fallback,
    and DuckDB / None as final honest fallback.
    """
    # 1. Primary: Official NSE MCP Server
    mcp_q = fetch_nse_mcp_stock_quote(sym_code, is_market_hours=is_mkt_open)
    if mcp_q and mcp_q.value is not None and mcp_q.value > 0:
        return yf_sym, mcp_q

    # 2. Secondary: Yahoo Finance fast_info
    try:
        t = yf.Ticker(yf_sym)
        fi = t.fast_info
        raw_last = getattr(fi, 'last_price', None)
        if raw_last is not None and float(raw_last) > 0:
            lp = float(raw_last)
            prev = float(getattr(fi, 'previous_close', lp) or lp)
            return yf_sym, make_quote(
                value=lp,
                ts_ist=datetime.now(IST),
                source="LIVE_YFINANCE",
                is_market_hours=is_mkt_open,
                open=float(getattr(fi, 'open', lp) or lp),
                high=float(getattr(fi, 'day_high', lp) or lp),
                low=float(getattr(fi, 'day_low', lp) or lp),
                prev=prev,
                is_real=True
            )
    except Exception:
        pass

    # 3. Final: Honest unavailable state (never fake numbers)
    return yf_sym, make_quote(
        value=None,
        ts_ist=datetime.now(IST),
        source="UNAVAILABLE",
        is_market_hours=is_mkt_open,
        is_real=False
    )


def fetch_live_market_snapshot(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Downloads live market snapshot across indices, heavyweights, and scanned stocks.
    Caches for CACHE_TTL_SECONDS to respect exchange rate limits.
    """
    now_ts = time.time()
    if not force_refresh and _CACHE["snapshot"] is not None and (now_ts - _CACHE["last_fetch_time"] < CACHE_TTL_SECONDS):
        return _CACHE["snapshot"]

    is_open, session_status, status_color = get_market_session_status()
    now_ist = datetime.now(IST)
    now_str = now_ist.strftime("%d %b %Y, %I:%M:%S %p IST")

    try:
        quote_map: Dict[str, Dict[str, Any]] = {}

        # 1. Batch fetch Indices & Global Catalysts
        index_quotes = fetch_batch_yfinance_quotes(INDEX_GLOBAL_TICKERS)
        for sym, q in index_quotes.items():
            quote_map[sym] = q.to_dict()

        # 2. Parallel fetch Equities (NSE MCP primary -> yfinance fallback)
        weights_map = load_index_weights_map()
        all_equities = {}
        for idx_stocks in weights_map.values():
            for s in idx_stocks:
                all_equities[s["sym"]] = s["yf"]
        all_equities.update(CHARTINK_STOCK_MAP)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_sym = {
                executor.submit(fetch_equity_quote, code, yf_sym, is_open): yf_sym
                for code, yf_sym in all_equities.items()
            }
            for future in concurrent.futures.as_completed(future_to_sym):
                try:
                    yf_sym, q = future.result()
                    quote_map[yf_sym] = q.to_dict()
                except Exception:
                    yf_sym = future_to_sym[future]
                    quote_map[yf_sym] = make_quote(None, source="FAILED", is_market_hours=is_open).to_dict()

        # 3. Build Index-Specific Constituents datasets from config/weights.csv
        constituents_by_index: Dict[str, List[Dict[str, Any]]] = {}
        for idx_name, idx_stocks in weights_map.items():
            stock_list = []
            for s in idx_stocks:
                q = quote_map.get(s["yf"], {})
                last_p = q.get("value")
                pct = q.get("pct_str", "N/A")
                bull = q.get("bull", True)
                chg_pct = q.get("chg_pct")

                if last_p is not None and chg_pct is not None:
                    price_str = f"₹{last_p:,.2f}"
                    if chg_pct > 1.0:
                        status = "⚡ Institutional Lead"
                    elif chg_pct > 0.0:
                        status = "🟢 Positive Accumulation"
                    elif chg_pct < -1.0:
                        status = "🔻 Heavy Drag"
                    else:
                        status = "🟡 Mild Pullback"
                else:
                    price_str = "N/A"
                    status = "⚪ Quote Stale/Offline"

                stock_list.append({
                    "sym": s["sym"],
                    "name": s["name"],
                    "wt": s["wt"],
                    "weight_pct": s.get("weight_pct", 0.0),
                    "sector": s.get("sector", ""),
                    "as_of_date": s.get("as_of_date", ""),
                    "price": price_str,
                    "pct": pct,
                    "chg_pct": chg_pct,
                    "status": status,
                    "bull": bull,
                    "source": q.get("source", "UNKNOWN"),
                    "is_stale": q.get("is_stale", True)
                })
            constituents_by_index[idx_name] = stock_list

        heavyweights_data = constituents_by_index.get("NIFTY 50", [])

        # 4. Chartink Equities Map
        chartink_live_quotes = {}
        for sym_code, yf_sym in CHARTINK_STOCK_MAP.items():
            chartink_live_quotes[sym_code] = quote_map.get(yf_sym, {})

        # 5. Fixed Floor Pivots (computed from PREVIOUS completed session in DuckDB)
        nifty_prev = get_previous_session_ohlc("NIFTY_SPOT")
        bank_prev = get_previous_session_ohlc("BANKNIFTY_SPOT")
        sensex_prev = get_previous_session_ohlc("SENSEX_SPOT")

        nifty_pivots = calc_floor_pivots(nifty_prev["high"] if nifty_prev else None, nifty_prev["low"] if nifty_prev else None, nifty_prev["close"] if nifty_prev else None)
        bank_pivots = calc_floor_pivots(bank_prev["high"] if bank_prev else None, bank_prev["low"] if bank_prev else None, bank_prev["close"] if bank_prev else None)
        sensex_pivots = calc_floor_pivots(sensex_prev["high"] if sensex_prev else None, sensex_prev["low"] if sensex_prev else None, sensex_prev["close"] if sensex_prev else None)

        n_q = quote_map.get("^NSEI", {})
        b_q = quote_map.get("^NSEBANK", {})
        s_q = quote_map.get("^BSESN", {})

        # 6. Post-15:30 IST Next-Day Pivots (from today's completed session candle)
        is_post_close = (now_ist.time() >= MARKET_CLOSE_TIME)
        nifty_next_pivots = calc_floor_pivots(n_q.get("high"), n_q.get("low"), n_q.get("last")) if is_post_close else None
        bank_next_pivots = calc_floor_pivots(b_q.get("high"), b_q.get("low"), b_q.get("last")) if is_post_close else None
        sensex_next_pivots = calc_floor_pivots(s_q.get("high"), s_q.get("low"), s_q.get("last")) if is_post_close else None

        # 7. Official GIFT Nifty
        gift_q = fetch_nse_gift_nifty(spot_nifty=n_q.get("last"))
        gift_basis = round((gift_q.get("last") or 0.0) - (n_q.get("last") or 0.0), 2) if (gift_q.get("last") and n_q.get("last")) else 0.0
        gift_prev = {
            "high": nifty_prev["high"] + gift_basis if (nifty_prev and nifty_prev.get("high")) else None,
            "low": nifty_prev["low"] + gift_basis if (nifty_prev and nifty_prev.get("low")) else None,
            "close": nifty_prev["close"] + gift_basis if (nifty_prev and nifty_prev.get("close")) else None
        }
        gift_pivots = calc_floor_pivots(gift_prev["high"], gift_prev["low"], gift_prev["close"])

        snapshot = {
            "timestamp": now_str,
            "ts_ist": now_ist,
            "is_open": is_open,
            "session_status": session_status,
            "status_color": status_color,
            "mcp_health": check_nse_mcp_health(),
            "mcp_stats": get_nse_mcp_stats(),
            "quote_map": quote_map,
            "heavyweights": heavyweights_data,
            "constituents_by_index": constituents_by_index,
            "chartink_live_quotes": chartink_live_quotes,
            "is_post_close": is_post_close,
            "nifty": {
                **n_q,
                **nifty_pivots,
                "next_day_pivots": nifty_next_pivots,
                "prev_session_date": nifty_prev["date"] if nifty_prev else "N/A"
            },
            "banknifty": {
                **b_q,
                **bank_pivots,
                "next_day_pivots": bank_next_pivots,
                "prev_session_date": bank_prev["date"] if bank_prev else "N/A"
            },
            "giftnifty": {
                **gift_q,
                **gift_pivots,
                "prev_session_date": nifty_prev["date"] if nifty_prev else "N/A"
            },
            "sensex": {
                **s_q,
                **sensex_pivots,
                "next_day_pivots": sensex_next_pivots,
                "prev_session_date": sensex_prev["date"] if sensex_prev else "N/A"
            },
            "vix": quote_map.get("^INDIAVIX", {}),
            "crude": quote_map.get("CL=F", {}),
            "usdinr": quote_map.get("USDINR=X", {}),
            "sp500": quote_map.get("ES=F", {}),
            "is_feed_live": sum(1 for q in quote_map.values() if q.get("is_live", False)) >= 3,
            "live_count": sum(1 for q in quote_map.values() if q.get("is_live", False)),
            "total_tickers": len(quote_map)
        }

        _CACHE["snapshot"] = snapshot
        _CACHE["last_fetch_time"] = now_ts
        return snapshot

    except Exception as e:
        print(f"Error in fetch_live_market_snapshot: {e}")
        if _CACHE["snapshot"] is not None:
            return _CACHE["snapshot"]
        raise e


def get_live_intraday_candles(symbol: str = "^NSEI", interval: str = "5m") -> pd.DataFrame:
    """
    Downloads genuine real-time intraday candles from Yahoo Finance.
    Caches results in memory for 60 seconds to prevent rate-limiting.
    If today's market is closed (e.g. overnight or weekend), pulls the most recent completed
    trading session's genuine 5m candles (period='5d').
    If 5m intraday is completely unavailable, falls back to DuckDB daily bars with explicit labeling.
    Never invents synthetic random/trigonometric numbers.
    """
    now_ts = time.time()
    cache_key = f"{symbol}_{interval}"
    cached = _CACHE.get("candles_5m", {}).get(cache_key)
    if cached and (now_ts - cached.get("time", 0.0) < 60.0):
        return cached["df"].copy()

    try:
        t = yf.Ticker(symbol)
        df = t.history(period="1d", interval=interval)
        if df.empty or len(df) < 5:
            # If 1d is empty or incomplete (weekend/overnight), fetch 5d and take latest session
            df = t.history(period="5d", interval=interval)
            if not df.empty:
                df = df.reset_index()
                dt_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
                df['TradeDate'] = pd.to_datetime(df[dt_col]).dt.date
                latest_trade_date = df['TradeDate'].max()
                df = df[df['TradeDate'] == latest_trade_date].drop(columns=['TradeDate'])

        if not df.empty:
            if "Datetime" not in df.columns and "Date" not in df.columns:
                df = df.reset_index()
            dt_col = "Datetime" if "Datetime" in df.columns else ("Date" if "Date" in df.columns else df.columns[0])
            df = df.rename(columns={dt_col: "Date"})
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
            if not df.empty:
                result_df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
                result_df.attrs["is_daily_fallback"] = False
                result_df.attrs["feed_type"] = "INTRADAY_5M_EXCHANGE"
                if "candles_5m" not in _CACHE:
                    _CACHE["candles_5m"] = {}
                _CACHE["candles_5m"][cache_key] = {"time": now_ts, "df": result_df.copy()}
                return result_df
    except Exception as e:
        print(f"Error fetching intraday candles for {symbol}: {e}")

    # Fallback to DuckDB historical daily prices (labeled: daily, not intraday)
    db_sym = SYMBOL_TO_DB_MAP.get(symbol, "NIFTY_SPOT")
    try:
        if os.path.exists(DB_PATH):
            con = duckdb.connect(DB_PATH, read_only=True)
            hist_df = con.execute("""
                SELECT date as Date, open as Open, high as High, low as Low, close as Close, volume as Volume 
                FROM daily_prices 
                WHERE symbol = ? 
                ORDER BY date DESC 
                LIMIT 30;
            """, [db_sym]).fetchdf()
            con.close()
            if not hist_df.empty:
                res = hist_df.sort_values("Date").reset_index(drop=True)
                res.attrs["is_daily_fallback"] = True
                res.attrs["feed_type"] = "DAILY_BAR_FALLBACK"
                return res
    except Exception as e:
        print(f"Error querying DuckDB daily fallback for {symbol} ({db_sym}): {e}")

    # Honest empty DataFrame on total failure (never invent synthetic price loops)
    empty_df = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    empty_df.attrs["is_daily_fallback"] = False
    empty_df.attrs["feed_type"] = "UNAVAILABLE"
    return empty_df
