"""
src/datapoint.py
Standardized Quantitative Market Datapoint & Quote Data Structure.
Enforces strict stale detection, IST timezone timestamps, data source provenance,
and safe fallback to None / "N/A" (never fake/static numbers).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any

IST = ZoneInfo("Asia/Kolkata")
STALE_THRESHOLD_SECONDS = 300.0  # 5 minutes during market hours


@dataclass
class Quote:
    """
    Standard market quote structure with explicit staleness and provenance tracking.
    """
    value: Optional[float]
    ts_ist: datetime
    source: str
    is_stale: bool
    age_sec: float

    # Extended OHLCV & change attributes
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    prev: Optional[float] = None
    chg_pts: Optional[float] = None
    chg_pct: Optional[float] = None
    volume: Optional[int] = None
    is_real: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Returns True only if value is present and not flagged as stale."""
        return self.value is not None and not self.is_stale

    @property
    def bull(self) -> bool:
        """True if price change is non-negative."""
        return (self.chg_pts or 0.0) >= 0.0

    def format_value(self, fmt: str = "{:,.2f}", fallback: str = "N/A") -> str:
        """Safe string formatting for quote value."""
        if self.value is None:
            return fallback
        return fmt.format(self.value)

    def format_pct(self, fallback: str = "N/A") -> str:
        """Safe string formatting for percentage change."""
        if self.chg_pct is None or self.value is None:
            return fallback
        sign = "+" if self.chg_pct >= 0 else ""
        return f"{sign}{self.chg_pct:.2f}%"

    def format_chg(self, fallback: str = "N/A") -> str:
        """Safe string formatting for point change."""
        if self.chg_pts is None or self.value is None:
            return fallback
        sign = "+" if self.chg_pts >= 0 else ""
        return f"{sign}{self.chg_pts:.2f}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to standard dictionary for dashboard backwards compatibility."""
        val = self.value
        p_str = self.format_pct(fallback="0.00%")
        c_str = self.format_chg(fallback="0.00")
        
        return {
            "last": val if val is not None else 0.0,
            "value": val,
            "open": self.open if self.open is not None else (val or 0.0),
            "high": self.high if self.high is not None else (val or 0.0),
            "low": self.low if self.low is not None else (val or 0.0),
            "prev": self.prev if self.prev is not None else (val or 0.0),
            "chg_pts": self.chg_pts if self.chg_pts is not None else 0.0,
            "chg_pct": self.chg_pct if self.chg_pct is not None else 0.0,
            "chg_str": c_str,
            "pct_str": p_str,
            "volume": self.volume or 0,
            "bull": self.bull,
            "ts_ist": self.ts_ist,
            "timestamp": self.ts_ist.strftime("%d %b %Y, %I:%M:%S %p IST"),
            "source": self.source,
            "is_stale": self.is_stale,
            "age_sec": round(self.age_sec, 1),
            "is_live": (not self.is_stale) and (val is not None) and (val > 0.0),
            "is_real": self.is_real,
            "metadata": self.metadata
        }


def make_quote(
    value: Optional[float],
    ts_ist: Optional[datetime] = None,
    source: str = "UNKNOWN",
    is_market_hours: bool = True,
    open: Optional[float] = None,
    high: Optional[float] = None,
    low: Optional[float] = None,
    prev: Optional[float] = None,
    chg_pts: Optional[float] = None,
    chg_pct: Optional[float] = None,
    volume: Optional[int] = None,
    is_real: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> Quote:
    """
    Constructs a Quote instance with calculated age_sec and strict staleness rule:
    Stale if value is None, or if during market hours data is older than 5 minutes (300s).
    """
    now_ist = datetime.now(IST)
    if ts_ist is None:
        ts_ist = now_ist
    elif ts_ist.tzinfo is None:
        ts_ist = ts_ist.replace(tzinfo=IST)
    else:
        ts_ist = ts_ist.astimezone(IST)

    age_sec = max(0.0, (now_ist - ts_ist).total_seconds())

    if value is None:
        is_stale = True
    elif is_market_hours and (age_sec > STALE_THRESHOLD_SECONDS):
        is_stale = True
    else:
        is_stale = False

    # Auto compute changes if not supplied
    if value is not None and prev is not None and prev > 0:
        if chg_pts is None:
            chg_pts = round(value - prev, 2)
        if chg_pct is None:
            chg_pct = round((chg_pts / prev) * 100.0, 2)

    return Quote(
        value=round(value, 2) if value is not None else None,
        ts_ist=ts_ist,
        source=source,
        is_stale=is_stale,
        age_sec=round(age_sec, 2),
        open=round(open, 2) if open is not None else None,
        high=round(high, 2) if high is not None else None,
        low=round(low, 2) if low is not None else None,
        prev=round(prev, 2) if prev is not None else None,
        chg_pts=round(chg_pts, 2) if chg_pts is not None else None,
        chg_pct=round(chg_pct, 2) if chg_pct is not None else None,
        volume=volume,
        is_real=is_real,
        metadata=metadata or {}
    )
