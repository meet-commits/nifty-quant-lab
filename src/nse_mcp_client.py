"""
src/nse_mcp_client.py
Official Client Adapter for National Stock Exchange of India (NSE) Model Context Protocol (MCP) Servers.
Connects directly to:
  - CM Live Market: https://mcp.nseindia.in/cmmkt/mcp
  - Historical Bhavcopy: https://mcp.nseindia.in/bhavcopy/cm/mcp

Uses JSON-RPC 2.0 over Streamable HTTP (SSE) protocol without any API keys or authentication.
100% Free & Open Exchange Interface.
"""

import json
import time
import requests
from typing import Dict, List, Any, Optional

CM_MKT_URL = "https://mcp.nseindia.in/cmmkt/mcp"
BHAVCOPY_URL = "https://mcp.nseindia.in/bhavcopy/cm/mcp"

_MCP_SESSION = None
_SESSION_ID = None
_SESSION_EXPIRY = 0.0
_QUOTE_CACHE: Dict[str, Dict[str, Any]] = {}
_LAST_MCP_CHECK = 0.0
_IS_MCP_ONLINE = False


def _parse_sse_json(text: str) -> Optional[Dict[str, Any]]:
    """Helper to parse SSE stream 'data: {...}' into a JSON dictionary."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
            try:
                return json.loads(payload)
            except Exception:
                pass
    return None


def _get_active_session() -> Optional[requests.Session]:
    """Establishes or reuses an active MCP JSON-RPC session handshake."""
    global _MCP_SESSION, _SESSION_ID, _SESSION_EXPIRY, _IS_MCP_ONLINE

    now = time.time()
    if _MCP_SESSION is not None and _SESSION_ID is not None and (now < _SESSION_EXPIRY):
        return _MCP_SESSION

    try:
        s = requests.Session()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream, */*",
            "User-Agent": "NiftyQuantLab/1.0 (Mozilla/5.0)"
        }
        init_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "nifty-quant-lab", "version": "1.0.0"}
            }
        }
        resp = s.post(CM_MKT_URL, headers=headers, json=init_payload, timeout=4)
        if resp.status_code == 200:
            sess_id = resp.headers.get("Mcp-Session-Id")
            if sess_id:
                _MCP_SESSION = s
                _SESSION_ID = sess_id
                _SESSION_EXPIRY = now + 300.0  # 5-minute session lifetime
                _IS_MCP_ONLINE = True
                return _MCP_SESSION
    except Exception as e:
        print(f"⚠️ NSE MCP Handshake Warning: {e}")

    _IS_MCP_ONLINE = False
    return None


def fetch_nse_mcp_stock_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Queries official NSE MCP server (cm_get_equity_stocks) for real-time stock quote.
    Returns normalized OHLCV, LTP, 52-week metrics, and change data.
    """
    clean_sym = symbol.replace(".NS", "").strip().upper()

    session = _get_active_session()
    if not session or not _SESSION_ID:
        return None

    try:
        call_payload = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000) % 100000,
            "method": "tools/call",
            "params": {
                "name": "cm_get_equity_stocks",
                "arguments": {
                    "symbolFilter": clean_sym,
                    "limit": 5
                }
            }
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream, */*",
            "Mcp-Session-Id": _SESSION_ID,
            "User-Agent": "NiftyQuantLab/1.0"
        }
        resp = session.post(CM_MKT_URL, headers=headers, json=call_payload, timeout=4)
        if resp.status_code == 200:
            parsed = _parse_sse_json(resp.text)
            if parsed and "result" in parsed:
                content = parsed["result"].get("content", [])
                if content and "text" in content[0]:
                    inner = json.loads(content[0]["text"])
                    stocks = inner.get("stocks", [])
                    for s in stocks:
                        if s.get("symbol") == clean_sym and s.get("series") == "EQ":
                            ltp = float(s.get("lastTradedPrice", 0.0))
                            prev = float(s.get("preClosePrice", ltp))
                            chg = float(s.get("change", ltp - prev))
                            pct = float(s.get("perChange", (chg / prev * 100.0) if prev else 0.0))
                            return {
                                "symbol": clean_sym,
                                "last": round(ltp, 2),
                                "prev": round(prev, 2),
                                "open": round(float(s.get("openPrice", ltp)), 2),
                                "high": round(float(s.get("highPrice", ltp)), 2),
                                "low": round(float(s.get("lowPrice", ltp)), 2),
                                "chg_pts": round(chg, 2),
                                "chg_pct": round(pct, 2),
                                "volume": int(s.get("volume", 0)),
                                "fiftyTwoWeekHigh": float(s.get("fiftyTwoWeekHigh", 0.0)),
                                "fiftyTwoWeekLow": float(s.get("fiftyTwoWeekLow", 0.0)),
                                "timestamp": s.get("latestTimestamp", ""),
                                "source": "OFFICIAL_NSE_MCP_SERVER",
                                "is_live": True
                            }
    except Exception as e:
        print(f"⚠️ Error querying NSE MCP for {clean_sym}: {e}")

    return None


def check_nse_mcp_health() -> Dict[str, Any]:
    """Returns connectivity and status information for official NSE MCP server."""
    sess = _get_active_session()
    return {
        "is_online": _IS_MCP_ONLINE,
        "endpoint": CM_MKT_URL,
        "transport": "Streamable HTTP (JSON-RPC 2.0 / SSE)",
        "session_id": _SESSION_ID if _SESSION_ID else "None",
        "official_provider": "National Stock Exchange of India (NSEIL)"
    }


if __name__ == "__main__":
    print("Testing Official NSE MCP Connection...")
    health = check_nse_mcp_health()
    print("Health:", health)
    if health["is_online"]:
        print("Querying RELIANCE via Official NSE MCP...")
        q = fetch_nse_mcp_stock_quote("RELIANCE")
        print("Quote:", q)
        print("Querying HDFCBANK via Official NSE MCP...")
        q2 = fetch_nse_mcp_stock_quote("HDFCBANK")
        print("Quote:", q2)
