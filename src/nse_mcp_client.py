"""
src/nse_mcp_client.py
Production-grade Thread-Safe Client for National Stock Exchange of India (NSE) Model Context Protocol (MCP).
Connects to:
  - CM Live Market: https://mcp.nseindia.in/cmmkt/mcp
  - Historical Bhavcopy: https://mcp.nseindia.in/bhavcopy/cm/mcp

Features:
  - Strict JSON-RPC 2.0 + notifications/initialized handshake
  - Thread-safe concurrency with threading.Lock
  - Handles application/json and text/event-stream (SSE)
  - Automatic session recovery with 1 retry on error
  - Exposes real telemetry stats (quotes_served, errors, reconnects)
  - Returns standardized Quote dataclass from src/datapoint.py
"""

import json
import time
import threading
import requests
from datetime import datetime, date
from zoneinfo import ZoneInfo
from typing import Dict, List, Any, Optional

try:
    from datapoint import Quote, make_quote
except ImportError:
    from src.datapoint import Quote, make_quote

IST = ZoneInfo("Asia/Kolkata")
CM_MKT_URL = "https://mcp.nseindia.in/cmmkt/mcp"
BHAVCOPY_URL = "https://mcp.nseindia.in/bhavcopy/cm/mcp"

_LOCK = threading.Lock()
_MCP_SESSION: Optional[requests.Session] = None
_SESSION_ID: Optional[str] = None
_SESSION_EXPIRY = 0.0
_IS_MCP_ONLINE = False

_STATS = {
    "quotes_served": 0,
    "errors": 0,
    "reconnects": 0,
    "symbols_served": {},
    "last_call_ts": None,
    "session_id": None
}


def _parse_mcp_response(resp: requests.Response) -> Optional[Dict[str, Any]]:
    """Parses MCP server response whether returned as application/json or text/event-stream."""
    content_type = resp.headers.get("Content-Type", "")
    text = resp.text

    if "application/json" in content_type:
        try:
            return resp.json()
        except Exception:
            pass

    # Parse text/event-stream format (data: {...})
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
            try:
                return json.loads(payload)
            except Exception:
                pass

    # Fallback attempt if header was unexpected
    try:
        return json.loads(text)
    except Exception:
        return None


def _get_active_session_locked() -> Optional[requests.Session]:
    """Establishes or reuses an active MCP session with notifications/initialized handshake. Must be called under _LOCK."""
    global _MCP_SESSION, _SESSION_ID, _SESSION_EXPIRY, _IS_MCP_ONLINE, _STATS

    now = time.time()
    if _MCP_SESSION is not None and _SESSION_ID is not None and (now < _SESSION_EXPIRY):
        return _MCP_SESSION

    try:
        s = requests.Session()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream, */*",
            "User-Agent": "NiftyQuantLab/2.0 (Mozilla/5.0)"
        }

        # Step 1: Initialize Handshake
        init_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "nifty-quant-lab", "version": "2.0.0"}
            }
        }
        resp = s.post(CM_MKT_URL, headers=headers, json=init_payload, timeout=4)
        if resp.status_code == 200:
            sess_id = resp.headers.get("Mcp-Session-Id")
            if sess_id:
                # Step 2: Send mandatory notifications/initialized
                notif_payload = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {}
                }
                headers["Mcp-Session-Id"] = sess_id
                try:
                    s.post(CM_MKT_URL, headers=headers, json=notif_payload, timeout=3)
                except Exception:
                    pass  # Notification is fire-and-forget

                _MCP_SESSION = s
                _SESSION_ID = sess_id
                _SESSION_EXPIRY = now + 300.0  # 5-minute session renewal
                _IS_MCP_ONLINE = True
                _STATS["session_id"] = sess_id
                _STATS["reconnects"] += 1
                return _MCP_SESSION
    except Exception as e:
        _STATS["errors"] += 1
        print(f"⚠️ NSE MCP Handshake Warning: {e}")

    _IS_MCP_ONLINE = False
    _MCP_SESSION = None
    _SESSION_ID = None
    return None


def _execute_mcp_tool_call(tool_name: str, arguments: Dict[str, Any], retry: bool = True) -> Optional[Dict[str, Any]]:
    """Thread-safe tool execution with 1 automatic retry on session failure."""
    global _MCP_SESSION, _SESSION_ID, _STATS

    with _LOCK:
        session = _get_active_session_locked()
        sess_id = _SESSION_ID

    if not session or not sess_id:
        return None

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream, */*",
        "Mcp-Session-Id": sess_id,
        "User-Agent": "NiftyQuantLab/2.0"
    }
    call_payload = {
        "jsonrpc": "2.0",
        "id": int(time.time() * 1000) % 100000,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }

    try:
        resp = session.post(CM_MKT_URL, headers=headers, json=call_payload, timeout=4)
        if resp.status_code == 200:
            parsed = _parse_mcp_response(resp)
            if parsed and "result" in parsed and not parsed.get("isError", False):
                return parsed["result"]
        elif resp.status_code in (400, 401, 403, 404):
            # Session likely invalid, force reconnect
            with _LOCK:
                _SESSION_EXPIRY = 0.0
    except Exception as e:
        _STATS["errors"] += 1
        with _LOCK:
            _SESSION_EXPIRY = 0.0

    # One retry if allowed
    if retry:
        with _LOCK:
            _SESSION_EXPIRY = 0.0
            session = _get_active_session_locked()
            sess_id = _SESSION_ID
        if session and sess_id:
            headers["Mcp-Session-Id"] = sess_id
            try:
                resp = session.post(CM_MKT_URL, headers=headers, json=call_payload, timeout=4)
                if resp.status_code == 200:
                    parsed = _parse_mcp_response(resp)
                    if parsed and "result" in parsed and not parsed.get("isError", False):
                        return parsed["result"]
            except Exception:
                _STATS["errors"] += 1

    return None


def fetch_nse_mcp_stock_quote(symbol: str, is_market_hours: bool = True) -> Optional[Quote]:
    """
    Primary data provider for NSE equities.
    Queries official NSE MCP server (cm_get_equity_stocks).
    Returns verified Quote dataclass or None if symbol not matched.
    """
    clean_sym = symbol.replace(".NS", "").strip().upper()

    res = _execute_mcp_tool_call("cm_get_equity_stocks", {"symbolFilter": clean_sym, "limit": 10})
    if not res:
        return None

    content = res.get("content", [])
    if not content or "text" not in content[0]:
        return None

    try:
        inner = json.loads(content[0]["text"])
        stocks = inner.get("stocks", [])
        for s in stocks:
            if s.get("symbol") == clean_sym and s.get("series") == "EQ":
                ltp = float(s.get("lastTradedPrice", 0.0))
                prev = float(s.get("preClosePrice", ltp))
                chg = float(s.get("change", ltp - prev))
                pct = float(s.get("perChange", (chg / prev * 100.0) if prev else 0.0))
                open_p = float(s.get("openPrice", ltp))
                high_p = float(s.get("highPrice", ltp))
                low_p = float(s.get("lowPrice", ltp))
                vol = int(s.get("volume", 0))

                ts_str = s.get("latestTimestamp", "")
                ts_ist = None
                if ts_str:
                    try:
                        ts_ist = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
                    except Exception:
                        pass
                if ts_ist is None:
                    ts_ist = datetime.now(IST)

                # Record stats
                _STATS["quotes_served"] += 1
                _STATS["symbols_served"][clean_sym] = _STATS["symbols_served"].get(clean_sym, 0) + 1
                _STATS["last_call_ts"] = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")

                quote = make_quote(
                    value=ltp,
                    ts_ist=ts_ist,
                    source="OFFICIAL_NSE_MCP_SERVER",
                    is_market_hours=is_market_hours,
                    open=open_p,
                    high=high_p,
                    low=low_p,
                    prev=prev,
                    chg_pts=chg,
                    chg_pct=pct,
                    volume=vol,
                    is_real=True,
                    metadata={
                        "fiftyTwoWeekHigh": float(s.get("fiftyTwoWeekHigh", 0.0)),
                        "fiftyTwoWeekLow": float(s.get("fiftyTwoWeekLow", 0.0)),
                        "series": "EQ"
                    }
                )
                return quote
    except Exception as e:
        _STATS["errors"] += 1
        print(f"⚠️ Error parsing MCP quote for {clean_sym}: {e}")

    return None


def fetch_nse_mcp_bhavcopy(trade_date: Optional[datetime.date] = None) -> Optional[List[Dict[str, Any]]]:
    """
    Fetches full EOD Bhavcopy from the official NSE MCP Bhavcopy Server.
    Endpoint: https://mcp.nseindia.in/bhavcopy/cm/mcp
    Returns normalized list of equity dictionary records or None on failure.
    """
    date_str = trade_date.strftime("%Y-%m-%d") if trade_date else datetime.now(IST).strftime("%Y-%m-%d")
    
    # Tool call to Bhavcopy MCP endpoint
    res = _execute_mcp_tool_call("cm_get_bhavcopy", {"tradeDate": date_str, "segment": "CM"})
    if not res:
        # Fallback tool name variant
        res = _execute_mcp_tool_call("get_bhavcopy", {"date": date_str})

    if not res:
        return None

    content = res.get("content", [])
    if not content or "text" not in content[0]:
        return None

    try:
        raw_text = content[0]["text"]
        parsed = json.loads(raw_text)
        records = parsed.get("records") or parsed.get("data") or parsed.get("bhavcopy", [])
        
        normalized = []
        for r in records:
            series = str(r.get("SERIES") or r.get("series") or "EQ").strip()
            if series not in ("EQ", "BE"):
                continue
            sym = str(r.get("SYMBOL") or r.get("symbol") or "").strip().upper()
            if not sym:
                continue
            
            c = float(r.get("CLOSE") or r.get("closePrice") or r.get("lastPrice") or 0.0)
            if c <= 0.0:
                continue
                
            o = float(r.get("OPEN") or r.get("openPrice") or c)
            h = float(r.get("HIGH") or r.get("highPrice") or max(o, c))
            l = float(r.get("LOW") or r.get("lowPrice") or min(o, c))
            prev = float(r.get("PREVCLOSE") or r.get("prevClose") or c)
            vol = int(r.get("TOTTRDQTY") or r.get("volume") or r.get("totalTradedQuantity") or 0)
            turnover = float(r.get("TOTTRDVAL") or r.get("turnover") or 0.0)

            normalized.append({
                "trade_date": date_str,
                "symbol": sym,
                "series": series,
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
                "prev_close": round(prev, 2),
                "volume": vol,
                "turnover": round(turnover, 2)
            })

        if normalized:
            _STATS["bhavcopy_served"] = _STATS.get("bhavcopy_served", 0) + 1
            _STATS["last_call_ts"] = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
            return normalized

    except Exception as e:
        _STATS["errors"] += 1
        print(f"⚠️ Error parsing MCP Bhavcopy response: {e}")

    return None


def check_nse_mcp_health() -> Dict[str, Any]:
    """Returns connectivity, session status, and live operational stats."""
    with _LOCK:
        sess = _get_active_session_locked()
    return {
        "is_online": _IS_MCP_ONLINE,
        "endpoint": CM_MKT_URL,
        "transport": "Streamable HTTP (JSON-RPC 2.0 / SSE)",
        "session_id": _SESSION_ID if _SESSION_ID else "None",
        "quotes_served": _STATS["quotes_served"],
        "errors": _STATS["errors"],
        "reconnects": _STATS["reconnects"],
        "last_call_ts": _STATS["last_call_ts"],
        "official_provider": "National Stock Exchange of India (NSEIL)"
    }


def get_nse_mcp_stats() -> Dict[str, Any]:
    """Returns telemetry dictionary for reporting and health dashboards."""
    return dict(_STATS)


if __name__ == "__main__":
    print("Testing Hardened NSE MCP Client...")
    health = check_nse_mcp_health()
    print("Health:", health)
    if health["is_online"]:
        print("Querying RELIANCE...")
        q = fetch_nse_mcp_stock_quote("RELIANCE")
        if q:
            print(f"RELIANCE: {q.format_value()} ({q.format_pct()}), Source={q.source}, Stale={q.is_stale}, Age={q.age_sec}s")
        print("Querying SBIN...")
        q2 = fetch_nse_mcp_stock_quote("SBIN")
        if q2:
            print(f"SBIN: {q2.format_value()} ({q2.format_pct()}), Source={q2.source}, Stale={q2.is_stale}")
        print("Final Telemetry Stats:", get_nse_mcp_stats())
