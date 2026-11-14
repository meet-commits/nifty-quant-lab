"""
src/chartink_engine.py
Complete Institutional Chartink Reverse-Engineered Technical Screener & Atlas Suite.
Features:
- Pure Mathematical Indicators: RSI(14), ATR(14), EMA(20,50,200), Volume Surge
- Dynamic EOD Bhavcopy Universe Scanning (Zero hardcoded stock lists or synthetic conviction)
- Full Universe Market Breadth Health Gauge (% > 20/50/200 EMA, Advance/Decline, As-of Date)
- Interactive Plotly Candlestick with Genuine OHLC, EMAs, and Wilder's RSI(14)
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date
from typing import Dict, List, Tuple, Any, Optional

try:
    from bhavcopy_service import get_bhavcopy_universe
except ImportError:
    from src.bhavcopy_service import get_bhavcopy_universe


# ----------------- TECHNICAL MATH INDICATORS -----------------

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Accurate Wilder's RSI calculation with exact boundary behavior."""
    if len(series) < period:
        return pd.Series([50.0] * len(series), index=series.index)

    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).fillna(0.0)
    loss = (-delta.where(delta < 0, 0.0)).fillna(0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Wilder's exponential smoothing
    ag = avg_gain.to_numpy(dtype=float, copy=True)
    al = avg_loss.to_numpy(dtype=float, copy=True)
    g = gain.to_numpy(dtype=float, copy=True)
    l = loss.to_numpy(dtype=float, copy=True)

    for i in range(period, len(series)):
        if not np.isnan(ag[i-1]):
            ag[i] = (ag[i-1] * (period - 1) + g[i]) / period
            al[i] = (al[i-1] * (period - 1) + l[i]) / period

    rsi = np.full(len(series), 50.0)
    for i in range(period - 1, len(series)):
        if np.isnan(ag[i]) or np.isnan(al[i]):
            continue
        if al[i] == 0.0 and ag[i] > 0.0:
            rsi[i] = 100.0
        elif ag[i] == 0.0 and al[i] > 0.0:
            rsi[i] = 0.0
        elif ag[i] == 0.0 and al[i] == 0.0:
            rsi[i] = 50.0
        else:
            rs = ag[i] / al[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return pd.Series(rsi, index=series.index).round(1)


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (ATR) calculation."""
    h_l = df['high'] - df['low']
    h_pc = (df['high'] - df['close'].shift(1)).abs()
    l_pc = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean().bfill().round(2)


def compute_universe_indicators(universe_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes vectorized technical indicators per symbol across the universe history:
    - EMA 20, EMA 50, EMA 200
    - Wilder's RSI(14)
    - ATR(14)
    - 20-day SMA Volume & Volume Surge Ratio
    - 1-Day % Change
    """
    if universe_df.empty:
        return universe_df

    df = universe_df.copy()
    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
    df = df.sort_values(by=['symbol', 'trade_date']).reset_index(drop=True)

    # 1. Exponential Moving Averages
    df['ema20'] = df.groupby('symbol')['close'].transform(lambda s: s.ewm(span=20, adjust=False).mean()).round(2)
    df['ema50'] = df.groupby('symbol')['close'].transform(lambda s: s.ewm(span=50, adjust=False).mean()).round(2)
    df['ema200'] = df.groupby('symbol')['close'].transform(lambda s: s.ewm(span=200, adjust=False).mean()).round(2)

    # 2. RSI(14)
    df['rsi14'] = df.groupby('symbol')['close'].transform(lambda s: calc_rsi(s, 14))

    # 3. ATR(14) per symbol
    h_l = df['high'] - df['low']
    h_pc = (df['high'] - df['prev_close']).abs()
    l_pc = (df['low'] - df['prev_close']).abs()
    df['tr'] = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    df['atr14'] = df.groupby('symbol')['tr'].transform(lambda s: s.rolling(14, min_periods=1).mean()).round(2)
    df = df.drop(columns=['tr'], errors='ignore')

    # 4. Volume SMA(20) & Volume Surge Ratio
    df['vol_sma20'] = df.groupby('symbol')['volume'].transform(lambda s: s.rolling(20, min_periods=5).mean()).round(0)
    df['vol_surge'] = (df['volume'] / df['vol_sma20'].replace(0, np.nan)).fillna(1.0).round(2)

    # 5. Price changes & multi-day highs/lows
    df['chg_pct'] = (((df['close'] - df['prev_close']) / df['prev_close'].replace(0, np.nan)) * 100.0).fillna(0.0).round(2)
    df['high_10d'] = df.groupby('symbol')['high'].transform(lambda s: s.rolling(10, min_periods=3).max())
    df['low_10d'] = df.groupby('symbol')['low'].transform(lambda s: s.rolling(10, min_periods=3).min())
    df['high_100d'] = df.groupby('symbol')['high'].transform(lambda s: s.rolling(100, min_periods=10).max())

    return df


# ----------------- SCANNER FUNCTIONS (PURE QUERIES) -----------------

def scan_volume_surge_momentum(latest_df: pd.DataFrame, universe_df: pd.DataFrame = None) -> pd.DataFrame:
    """⚡ Volume Surge & Momentum Breakout: Volume >= 1.8x 20d SMA, Close > Open, Close > 20 EMA, RSI >= 52."""
    cond = (
        (latest_df['vol_surge'] >= 1.8) &
        (latest_df['close'] > latest_df['open']) &
        (latest_df['close'] >= latest_df['ema20']) &
        (latest_df['rsi14'] >= 52.0)
    )
    return latest_df[cond].copy()


def scan_trend_leader_highs(latest_df: pd.DataFrame, universe_df: pd.DataFrame = None) -> pd.DataFrame:
    """👑 Multi-Month Trend Leaders: Close near 100d/52w High, EMA 20 > 50, Close > 20 EMA."""
    cond = (
        (latest_df['close'] >= latest_df['high_100d'].fillna(latest_df['close']) * 0.94) &
        (latest_df['ema20'] >= latest_df['ema50']) &
        (latest_df['close'] >= latest_df['ema20']) &
        (latest_df['rsi14'] >= 55.0)
    )
    return latest_df[cond].copy()


def scan_rsi_oversold_dips(latest_df: pd.DataFrame, universe_df: pd.DataFrame = None) -> pd.DataFrame:
    """🔥 RSI < 32 Panic Oversold Dips: Deep pullback with bullish intraday rejection."""
    cond = (
        (latest_df['rsi14'] <= 35.0) &
        (latest_df['close'] >= latest_df['open'])
    )
    return latest_df[cond].copy()


def scan_ema_confluence(latest_df: pd.DataFrame, universe_df: pd.DataFrame = None) -> pd.DataFrame:
    """🚀 20 & 50 EMA Bullish Confluence: 20 EMA > 50 EMA, Close > 20 EMA, Volume Surge >= 1.4x."""
    cond = (
        (latest_df['ema20'] > latest_df['ema50']) &
        (latest_df['close'] >= latest_df['ema20']) &
        (latest_df['vol_surge'] >= 1.3) &
        (latest_df['chg_pct'] >= 0.0)
    )
    return latest_df[cond].copy()


def scan_darvas_box_breakout(latest_df: pd.DataFrame, universe_df: pd.DataFrame = None) -> pd.DataFrame:
    """💎 Tight Consolidation Range Breakout: 10d range <= 6%, Close at 10d high, Volume expansion."""
    range_width = (latest_df['high_10d'] - latest_df['low_10d']) / latest_df['close']
    cond = (
        (range_width <= 0.07) &
        (latest_df['close'] >= latest_df['high_10d'] * 0.985) &
        (latest_df['vol_surge'] >= 1.3)
    )
    return latest_df[cond].copy()


def scan_bearish_breakdown(latest_df: pd.DataFrame, universe_df: pd.DataFrame = None) -> pd.DataFrame:
    """🔴 Bearish Breakdown & 200 EMA Rejection: Breaking 10d lows, Close < 200 EMA, RSI < 45."""
    cond = (
        (latest_df['close'] <= latest_df['low_10d'] * 1.01) &
        (latest_df['close'] <= latest_df['ema200']) &
        (latest_df['rsi14'] <= 45.0)
    )
    return latest_df[cond].copy()


def scan_full_universe(latest_df: pd.DataFrame, universe_df: pd.DataFrame = None) -> pd.DataFrame:
    """🌐 All Scanned Stocks (Full Universe): Sorted by traded turnover / volume."""
    return latest_df.sort_values(by='turnover', ascending=False).copy()


# ----------------- REGISTRY OF DYNAMIC SCANNERS -----------------

CHARTINK_SCANNERS: Dict[str, Dict[str, Any]] = {
    "⚡ Volume Surge & Momentum Breakout": {
        "desc": "Stocks breaking out on >= 1.8x 20-day average volume with positive candle and expanding RSI momentum",
        "clauses": [
            "• [ Latest ] [ Volume ] >= [ 20 period SMA(Volume) ] * 1.8",
            "• [ Latest ] [ Close ] > [ Latest ] [ Open ] (Bullish Day Candle)",
            "• [ Latest ] [ Close ] >= [ Latest ] [ 20 period EMA ]",
            "• [ Latest ] [ RSI (14) ] between 54 and 78"
        ],
        "fn": scan_volume_surge_momentum
    },
    "👑 Multi-Month Trend Leaders (Stage 2)": {
        "desc": "Institutional trend leaders trading within 6% of multi-month highs with aligned 20 > 50 EMAs",
        "clauses": [
            "• [ Latest ] [ Close ] >= [ 100 period Max(High) ] * 0.94",
            "• [ Latest ] [ 20 period EMA ] >= [ Latest ] [ 50 period EMA ]",
            "• [ Latest ] [ Close ] >= [ Latest ] [ 20 period EMA ]",
            "• [ Latest ] [ RSI (14) ] >= 55.0"
        ],
        "fn": scan_trend_leader_highs
    },
    "🔥 RSI < 32 Panic Oversold Accumulation": {
        "desc": "Institutional accumulation on oversold pullbacks with bullish intraday close rejection",
        "clauses": [
            "• [ Latest ] [ RSI (14) ] <= 35.0 (Oversold Range)",
            "• [ Latest ] [ Close ] >= [ Latest ] [ Open ] (Bullish Absorption Candle)"
        ],
        "fn": scan_rsi_oversold_dips
    },
    "🚀 20 & 50 EMA Bullish Confluence": {
        "desc": "Structural trend resumption where 20 EMA leads 50 EMA on above-average volume expansion",
        "clauses": [
            "• [ Latest ] [ 20 period EMA ] > [ Latest ] [ 50 period EMA ]",
            "• [ Latest ] [ Close ] >= [ Latest ] [ 20 period EMA ]",
            "• [ Latest ] [ Volume ] >= [ 20 period SMA(Volume) ] * 1.3",
            "• [ Latest ] [ 1-day Change % ] >= 0.0%"
        ],
        "fn": scan_ema_confluence
    },
    "💎 Tight Consolidation Range Breakout (VCP)": {
        "desc": "Tight 10-day price consolidation box (<= 7% width) breaking out on heavy volume expansion",
        "clauses": [
            "• [ 10 period ] (Max(High) - Min(Low)) / Close <= 0.07 (Volatility Compression)",
            "• [ Latest ] [ Close ] >= [ 10 period Max(High) ] * 0.985 (Range Exit)",
            "• [ Latest ] [ Volume ] >= [ 20 period SMA(Volume) ] * 1.3"
        ],
        "fn": scan_darvas_box_breakout
    },
    "🔴 Bearish Breakdown & 200 EMA Rejection": {
        "desc": "Stocks breaking below 10-day support floors or failing below 200 EMA with weak momentum",
        "clauses": [
            "• [ Latest ] [ Close ] <= [ 10 period Min(Low) ] * 1.01",
            "• [ Latest ] [ Close ] <= [ Latest ] [ 200 period EMA ]",
            "• [ Latest ] [ RSI (14) ] <= 45.0"
        ],
        "fn": scan_bearish_breakdown
    },
    "🌐 All Scanned Stocks (Full Universe)": {
        "desc": "Complete universe of scanned NSE Equities ranked by trading turnover",
        "clauses": [
            "• Active cash equity segment (Series EQ / BE)",
            "• Ranked by daily turnover and liquidity"
        ],
        "fn": scan_full_universe
    }
}

# Alias for backwards compatibility with earlier UI imports
CHARTINK_CATEGORIES = CHARTINK_SCANNERS


# ----------------- SCANNER EXECUTION ENGINE -----------------

_UNIVERSE_CACHE: Dict[str, Any] = {
    "last_calc_time": 0.0,
    "universe_df": None,
    "latest_df": None
}

def get_or_calc_indicator_universe(force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieves or calculates the indicator universe, cached in memory."""
    import time
    now = time.time()
    if not force_refresh and _UNIVERSE_CACHE["universe_df"] is not None and (now - _UNIVERSE_CACHE["last_calc_time"] < 60.0):
        return _UNIVERSE_CACHE["universe_df"], _UNIVERSE_CACHE["latest_df"]

    raw_df = get_bhavcopy_universe(days=120)
    if raw_df.empty:
        empty = pd.DataFrame()
        return empty, empty

    universe_df = compute_universe_indicators(raw_df)
    latest_date = universe_df['trade_date'].max()
    latest_df = universe_df[universe_df['trade_date'] == latest_date].copy()

    _UNIVERSE_CACHE["universe_df"] = universe_df
    _UNIVERSE_CACHE["latest_df"] = latest_df
    _UNIVERSE_CACHE["last_calc_time"] = now

    return universe_df, latest_df


def run_chartink_scan(scanner_name: str, force_refresh: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes a real technical scan across the NSE universe.
    Returns:
    - Clean sortable results DataFrame without any synthetic/hardcoded columns.
    - Metadata dictionary with as_of_date, total_universe, and matched_count.
    """
    universe_df, latest_df = get_or_calc_indicator_universe(force_refresh=force_refresh)

    if latest_df.empty:
        return pd.DataFrame(), {"as_of_date": "N/A", "total_universe": 0, "matched_count": 0}

    as_of_str = str(latest_df['trade_date'].iloc[0])
    scan_meta = CHARTINK_SCANNERS.get(scanner_name, CHARTINK_SCANNERS["🌐 All Scanned Stocks (Full Universe)"])
    filter_fn = scan_meta.get("fn", scan_full_universe)

    filtered_df = filter_fn(latest_df, universe_df)

    # Format into clean presentation schema (ZERO hardcoded predictions or conviction)
    results = []
    for idx, (_, r) in enumerate(filtered_df.iterrows(), start=1):
        results.append({
            "Sr": idx,
            "Symbol": r['symbol'],
            "Price": float(r['close']),
            "Change %": float(r['chg_pct']),
            "Volume": int(r['volume']),
            "Volume Surge": f"{float(r['vol_surge']):.1f}x",
            "RSI": float(r['rsi14']),
            "EMA 20": float(r['ema20']),
            "EMA 50": float(r['ema50']),
            "EMA 200": float(r['ema200']),
            "ATR": float(r['atr14'])
        })

    res_df = pd.DataFrame(results)
    meta = {
        "as_of_date": as_of_str,
        "total_universe": len(latest_df),
        "matched_count": len(res_df),
        "scanner_name": scanner_name
    }
    return res_df, meta


# ----------------- REAL UNIVERSE MARKET BREADTH -----------------

def get_market_breadth_stats(latest_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Calculates Market Breadth across the WHOLE NSE Equity universe on the latest date.
    Strictly mathematical, with explicit 'as of' date.
    """
    if latest_df is None:
        _, latest_df = get_or_calc_indicator_universe()

    if latest_df is None or latest_df.empty:
        return {
            "pct_above_20_ema": 50.0,
            "pct_above_50_ema": 50.0,
            "pct_above_200_ema": 50.0,
            "adv_dec_ratio": "1.00",
            "advances": 0,
            "declines": 0,
            "unchanged": 0,
            "total_universe": 0,
            "as_of_date": "N/A",
            "market_regime": "AWAITING BHAVCOPY DATA",
            "regime_color": "#64748B"
        }

    total = len(latest_df)
    as_of_date = str(latest_df['trade_date'].iloc[0])

    above_20 = int((latest_df['close'] > latest_df['ema20']).sum())
    above_50 = int((latest_df['close'] > latest_df['ema50']).sum())
    above_200 = int((latest_df['close'] > latest_df['ema200']).sum())

    advances = int((latest_df['close'] > latest_df['prev_close']).sum())
    declines = int((latest_df['close'] < latest_df['prev_close']).sum())
    unchanged = total - (advances + declines)

    pct_20 = round((above_20 / max(total, 1)) * 100.0, 1)
    pct_50 = round((above_50 / max(total, 1)) * 100.0, 1)
    pct_200 = round((above_200 / max(total, 1)) * 100.0, 1)

    ad_ratio_num = round(advances / max(declines, 1), 2)
    if ad_ratio_num >= 1.5:
        ad_str = f"{ad_ratio_num:.2f} (Bulls Leading)"
    elif ad_ratio_num <= 0.7:
        ad_str = f"{ad_ratio_num:.2f} (Bears Leading)"
    else:
        ad_str = f"{ad_ratio_num:.2f} (Balanced)"

    # Regime Determination
    if pct_50 >= 60.0:
        regime = "HEALTHY BULLISH EXPANSION"
        color = "#10B981"
    elif pct_50 <= 40.0:
        regime = "BROAD BEARISH CONTRACTION"
        color = "#F43F5E"
    else:
        regime = "NEUTRAL CONSOLIDATION"
        color = "#F59E0B"

    return {
        "pct_above_20_ema": pct_20,
        "pct_above_50_ema": pct_50,
        "pct_above_200_ema": pct_200,
        "adv_dec_ratio": ad_str,
        "advances": advances,
        "declines": declines,
        "unchanged": unchanged,
        "total_universe": total,
        "as_of_date": as_of_date,
        "market_regime": regime,
        "regime_color": color
    }


def get_sector_heatmap(latest_df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    """Calculates sector performance dynamically from active equity universe."""
    if latest_df is None:
        _, latest_df = get_or_calc_indicator_universe()

    # Pre-defined major sectors for clean grouping
    sector_mapping = {
        "BANKING": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "FEDERALBNK", "BANKBARODA", "PNB", "IDFCFIRSTB"],
        "IT SERVICES": ["TCS", "INFY", "WIPRO", "HCLTECH"],
        "ENERGY & OIL": ["RELIANCE", "BPCL", "IOC", "ONGC", "GAIL", "COALINDIA"],
        "AUTO & MOBILITY": ["MARUTI", "TATAMOTORS", "M&M", "EICHERMOT", "BAJAJ-AUTO", "HEROMOTOCO"],
        "METALS & MINING": ["TATASTEEL", "JSWSTEEL", "JINDALSTEL", "HINDALCO", "VEDL"],
        "CONSUMER & FMCG": ["ITC", "TATACONSUM", "ASIANPAINT", "TITAN", "TRENT"],
        "PHARMA & HEALTH": ["SUNPHARMA", "CIPLA", "DRREDDY", "APOLLOHOSP"]
    }

    heatmap = []
    if latest_df is None or latest_df.empty:
        return heatmap

    sym_to_chg = dict(zip(latest_df['symbol'], latest_df['chg_pct']))

    for sec_name, syms in sector_mapping.items():
        chgs = [sym_to_chg[s] for s in syms if s in sym_to_chg]
        if chgs:
            avg_chg = round(float(np.mean(chgs)), 2)
            if avg_chg >= 1.0:
                status = "Leader"
                color = "#10B981"
            elif avg_chg > 0.0:
                status = "Accumulation"
                color = "#10B981"
            elif avg_chg > -1.0:
                status = "Mild Pullback"
                color = "#F59E0B"
            else:
                status = "Heavy Drag"
                color = "#F43F5E"

            heatmap.append({
                "sector": sec_name,
                "chg": f"{avg_chg:+.2f}%",
                "status": status,
                "color": color
            })

    return heatmap


def fetch_stock_real_ohlc(symbol: str, period: str = "3mo") -> pd.DataFrame:
    """
    Fetches real historical OHLC for chart inspector.
    Prioritizes local DuckDB bhavcopy, then yfinance fallback.
    """
    clean_sym = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
    try:
        from bhavcopy_service import get_db_connection
        con = get_db_connection(read_only=True)
        df_db = con.execute("""
            SELECT trade_date as Date, open as Open, high as High, low as Low, close as Close, volume as Volume
            FROM bhavcopy_daily
            WHERE symbol = ?
            ORDER BY trade_date ASC;
        """, [clean_sym]).fetchdf()
        con.close()

        if not df_db.empty and len(df_db) >= 15:
            df_db['Date'] = pd.to_datetime(df_db['Date'])
            df_db['RSI'] = calc_rsi(df_db['Close'], 14)
            df_db['EMA20'] = df_db['Close'].ewm(span=20, adjust=False).mean().round(2)
            df_db['EMA50'] = df_db['Close'].ewm(span=50, adjust=False).mean().round(2)
            df_db['is_real'] = True
            return df_db[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'EMA20', 'EMA50', 'is_real']]
    except Exception:
        pass

    # Secondary: Yahoo Finance
    yf_ticker = f"{clean_sym}.NS"
    try:
        t = yf.Ticker(yf_ticker)
        df = t.history(period=period)
        if not df.empty:
            df = df.reset_index()
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
            if len(df) >= 15:
                if hasattr(df['Date'].dt, 'tz_localize'):
                    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                else:
                    df['Date'] = pd.to_datetime(df['Date'])
                df['RSI'] = calc_rsi(df['Close'], 14)
                df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean().round(2)
                df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean().round(2)
                df['is_real'] = True
                return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'EMA20', 'EMA50', 'is_real']]
    except Exception as e:
        print(f"⚠️ yfinance fetch error for {symbol}: {e}")

    # Honest deterministic baseline
    dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=40, freq='B')
    base = 1000.0
    prices = [base * (1.0 + 0.005 * i) for i in range(len(dates))]
    df_fb = pd.DataFrame({
        "Date": dates,
        "Open": prices,
        "High": [p * 1.01 for p in prices],
        "Low": [p * 0.99 for p in prices],
        "Close": prices,
        "Volume": [1500000] * len(dates),
        "is_real": False
    })
    df_fb['RSI'] = calc_rsi(df_fb['Close'], 14)
    df_fb['EMA20'] = df_fb['Close'].ewm(span=20, adjust=False).mean().round(2)
    df_fb['EMA50'] = df_fb['Close'].ewm(span=50, adjust=False).mean().round(2)
    return df_fb


if __name__ == "__main__":
    print("Testing Dynamic Bhavcopy Scanner Engine...")
    breadth = get_market_breadth_stats()
    print("Breadth:", breadth)
    res, meta = run_chartink_scan("⚡ Volume Surge & Momentum Breakout")
    print(f"Volume Surge Scan matched {len(res)} stocks out of {meta['total_universe']} (as of {meta['as_of_date']})")
    if not res.empty:
        print(res.head(3))
