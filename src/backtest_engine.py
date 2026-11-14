"""
src/backtest_engine.py
Purpose: Realistic Black-Scholes Options Backtesting & Break-Even Evaluation Engine.
Features:
- Dynamic Black-Scholes option pricing (ATM strike, India VIX as IV)
- Calendar-day overnight theta decay (3 calendar days over weekends vs 1 day on weekdays)
- Non-linear option revaluation at open (incorporating Gamma and Delta)
- Statutory Indian taxes & friction loaded from config/charges.yaml
- Break-Even Analysis: Computes exact index gap points required to overcome theta + friction.
"""

import os
import yaml
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Any, Tuple, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHARGES_PATH = os.path.join(BASE_DIR, "config", "charges.yaml")


def load_charges_config() -> Dict[str, Any]:
    """Loads statutory transaction costs and tax rates from config/charges.yaml."""
    default_config = {
        "equity_options": {
            "stt_sell_turnover": 0.0010,
            "exchange_turnover": 0.0005,
            "gst_rate": 0.18,
            "stamp_duty_buy": 0.00003,
            "slippage_pct": 0.015,
            "nifty_lot_size": 25
        },
        "risk_free_rate": 0.065
    }
    if os.path.exists(CHARGES_PATH):
        try:
            with open(CHARGES_PATH, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if cfg:
                    return cfg
        except Exception as e:
            print(f"⚠️ Error reading {CHARGES_PATH}: {e}. Using statutory defaults.")
    return default_config


def black_scholes_price(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    iv: float,
    option_type: str = "call"
) -> float:
    """
    Computes theoretical European option price via classical Black-Scholes-Merton formula.
    """
    s = max(float(spot), 1.0)
    k = max(float(strike), 1.0)
    t = max(float(time_years), 0.0005)  # Minimum ~4 hours to expiration
    sigma = max(float(iv), 0.05)       # Minimum 5% IV floor
    r = float(rate)

    d1 = (math.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)

    if option_type.lower() == "call":
        price = s * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)
    else:
        price = k * math.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)

    return max(float(price), 0.05)


def black_scholes_greeks(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    iv: float
) -> Dict[str, float]:
    """
    Computes theoretical Black-Scholes Greeks: Call/Put Delta, Gamma, Vega, Theta.
    Theta is returned in points per calendar day.
    """
    s = max(float(spot), 1.0)
    k = max(float(strike), 1.0)
    t = max(float(time_years), 0.0005)
    sigma = max(float(iv), 0.05)
    r = float(rate)

    d1 = (math.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)

    nd1 = norm.cdf(d1)
    np_d1 = norm.pdf(d1)
    nd2 = norm.cdf(d2)

    call_delta = nd1
    put_delta = nd1 - 1.0
    gamma = np_d1 / (s * sigma * math.sqrt(t))
    vega = s * np_d1 * math.sqrt(t) / 100.0  # per 1% change in IV

    # Annual Theta
    annual_term1 = -(s * np_d1 * sigma) / (2.0 * math.sqrt(t))
    call_theta_annual = annual_term1 - r * k * math.exp(-r * t) * nd2
    put_theta_annual = annual_term1 + r * k * math.exp(-r * t) * norm.cdf(-d2)

    # Convert to points per calendar day (365 days)
    call_theta_per_day = call_theta_annual / 365.0
    put_theta_per_day = put_theta_annual / 365.0

    return {
        "call_delta": round(float(call_delta), 4),
        "put_delta": round(float(put_delta), 4),
        "gamma": round(float(gamma), 6),
        "vega": round(float(vega), 4),
        "call_theta_daily": round(float(call_theta_per_day), 2),
        "put_theta_daily": round(float(put_theta_per_day), 2)
    }


def compute_option_breakeven(
    spot: float,
    vix: float,
    days_to_expiry: float = 3.0,
    calendar_days_held: int = 1,
    option_type: str = "call"
) -> Dict[str, Any]:
    """
    Computes exact index gap points required to overcome overnight theta decay and transaction costs.
    """
    cfg = load_charges_config()
    eq_cfg = cfg.get("equity_options", {})
    r = cfg.get("risk_free_rate", 0.065)

    strike = round(spot / 50.0) * 50.0
    iv = max(float(vix or 15.0) / 100.0, 0.08)
    t_years = max(days_to_expiry, 0.5) / 365.0

    entry_premium = black_scholes_price(spot, strike, t_years, r, iv, option_type)
    greeks = black_scholes_greeks(spot, strike, t_years, r, iv)
    
    daily_theta = abs(greeks["call_theta_daily"] if option_type.lower() == "call" else greeks["put_theta_daily"])
    overnight_theta_loss = daily_theta * calendar_days_held

    # Transaction friction per contract
    slippage = eq_cfg.get("slippage_pct", 0.015)
    stt_rate = eq_cfg.get("stt_sell_turnover", 0.0010)
    exch_rate = eq_cfg.get("exchange_turnover", 0.0005)
    gst_rate = eq_cfg.get("gst_rate", 0.18)
    stamp_rate = eq_cfg.get("stamp_duty_buy", 0.00003)

    # Estimate friction in option premium points
    est_friction_pts = (
        (entry_premium * slippage * 2.0) +
        (entry_premium * stt_rate) +
        (entry_premium * 2.0 * exch_rate * (1.0 + gst_rate)) +
        (entry_premium * stamp_rate)
    )

    total_premium_loss_pts = overnight_theta_loss + est_friction_pts
    delta = abs(greeks["call_delta"] if option_type.lower() == "call" else greeks["put_delta"])
    delta = max(delta, 0.20)

    # Required underlying move in index points to break even
    breakeven_gap_pts = round(total_premium_loss_pts / delta, 1)

    return {
        "spot": spot,
        "atm_strike": strike,
        "entry_premium": round(entry_premium, 1),
        "overnight_theta_loss_pts": round(overnight_theta_loss, 1),
        "friction_pts": round(est_friction_pts, 1),
        "delta": round(delta, 3),
        "breakeven_gap_pts": breakeven_gap_pts,
        "calendar_days_held": calendar_days_held,
        "vix_iv_pct": round(iv * 100.0, 1)
    }


def calculate_transaction_friction(
    entry_premium: float,
    exit_premium: float,
    lot_size: int,
    cfg: Dict[str, Any]
) -> float:
    """Calculates comprehensive statutory Indian friction (STT, NSE, GST, Stamp, Slippage)."""
    eq = cfg.get("equity_options", {})
    slippage_pct = eq.get("slippage_pct", 0.015)
    stt_rate = eq.get("stt_sell_turnover", 0.0010)
    exch_rate = eq.get("exchange_turnover", 0.0005)
    gst_rate = eq.get("gst_rate", 0.18)
    stamp_rate = eq.get("stamp_duty_buy", 0.00003)

    buy_turnover = entry_premium * lot_size
    sell_turnover = exit_premium * lot_size

    slippage_cost = ((entry_premium * slippage_pct) + (exit_premium * slippage_pct)) * lot_size
    stt = sell_turnover * stt_rate
    exchange_charges = (buy_turnover + sell_turnover) * exch_rate
    gst = exchange_charges * gst_rate
    stamp_duty = buy_turnover * stamp_rate

    return slippage_cost + stt + exchange_charges + gst + stamp_duty


def run_realistic_backtest(
    predictions_df: pd.DataFrame,
    initial_capital: float = 100000.0,
    prob_threshold_high: float = 0.55,
    prob_threshold_low: float = 0.45,
    simulated_sl_pct: float = 0.30
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes an out-of-sample Options Backtest using dynamic Black-Scholes pricing
    and calendar-day theta decay loaded with real statutory charges from YAML.
    """
    cfg = load_charges_config()
    lot_size = cfg.get("equity_options", {}).get("nifty_lot_size", 25)
    r = cfg.get("risk_free_rate", 0.065)

    capital = initial_capital
    peak_capital = initial_capital
    max_drawdown_pct = 0.0
    trade_records = []

    for i in range(len(predictions_df)):
        row = predictions_df.iloc[i]
        prob_up = float(row.get('predicted_prob_up', 0.50))
        prob_down = float(row.get('predicted_prob_down', 0.0))
        nifty_spot = float(row['nifty_close'])
        actual_gap_pct = float(row['actual_gap_pct'])
        actual_gap_pts = (actual_gap_pct / 100.0) * nifty_spot

        # Determine trading signal
        if prob_up >= prob_threshold_high:
            signal = "CALL"
        elif prob_down >= prob_threshold_low or (prob_up <= (1.0 - prob_threshold_high) and prob_down == 0.0):
            signal = "PUT"
        else:
            signal = "NO_TRADE"

        if signal == "NO_TRADE":
            continue

        # Calendar days to next session (3 over weekend, 1 on weekdays)
        row_date = pd.to_datetime(row['date'])
        cal_days = 3 if row_date.weekday() == 4 else 1  # Friday -> Monday
        dte = max(float(row.get('days_to_expiry', 3)), 1.0)
        t_entry = dte / 365.0
        t_exit = max(dte - cal_days, 0.25) / 365.0

        vix_val = float(row.get('vix_close', 15.0) or 15.0)
        iv = max(vix_val / 100.0, 0.08)
        strike = round(nifty_spot / 50.0) * 50.0

        opt_type = "call" if signal == "CALL" else "put"
        entry_premium = black_scholes_price(nifty_spot, strike, t_entry, r, iv, opt_type)

        # Revalue option at tomorrow's opening spot price
        next_open_spot = nifty_spot + actual_gap_pts
        theoretical_exit = black_scholes_price(next_open_spot, strike, t_exit, r, iv, opt_type)

        # Enforce stop loss boundary
        min_allowed_premium = entry_premium * (1.0 - simulated_sl_pct)
        exit_premium = max(theoretical_exit, min_allowed_premium)
        exit_premium = max(exit_premium, 2.0)

        # Calculate friction and P&L
        friction = calculate_transaction_friction(entry_premium, exit_premium, lot_size, cfg)
        gross_pnl = (exit_premium - entry_premium) * lot_size
        net_pnl = gross_pnl - friction

        capital += net_pnl
        if capital > peak_capital:
            peak_capital = capital
        drawdown = (peak_capital - capital) / peak_capital * 100.0
        if drawdown > max_drawdown_pct:
            max_drawdown_pct = drawdown

        trade_records.append({
            "date": row_date.strftime('%Y-%m-%d'),
            "signal": signal,
            "probability": round(prob_up if signal == "CALL" else prob_down, 3),
            "actual_gap_pct": round(actual_gap_pct, 2),
            "actual_gap_pts": round(actual_gap_pts, 1),
            "entry_premium": round(entry_premium, 2),
            "exit_premium": round(exit_premium, 2),
            "gross_pnl": round(gross_pnl, 2),
            "friction_cost": round(friction, 2),
            "net_pnl": round(net_pnl, 2),
            "capital_after_trade": round(capital, 2),
            "drawdown_pct": round(drawdown, 2)
        })

    trades_df = pd.DataFrame(trade_records)

    if not trades_df.empty:
        winning_trades = trades_df[trades_df['net_pnl'] > 0]
        losing_trades = trades_df[trades_df['net_pnl'] <= 0]
        win_rate = len(winning_trades) / len(trades_df) * 100.0
        gross_wins = winning_trades['net_pnl'].sum()
        gross_losses = abs(losing_trades['net_pnl'].sum())
        profit_factor = round(gross_wins / gross_losses, 2) if gross_losses > 0 else np.nan
        total_return_pct = round(((capital - initial_capital) / initial_capital) * 100.0, 2)
        total_friction_paid = round(trades_df['friction_cost'].sum(), 2)
    else:
        win_rate = 0.0
        profit_factor = 0.0
        total_return_pct = 0.0
        total_friction_paid = 0.0

    summary = {
        "starting_capital": initial_capital,
        "final_capital": round(capital, 2),
        "total_filtered_trades": len(trades_df),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": profit_factor,
        "net_return_pct": total_return_pct,
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "total_friction_paid_inr": total_friction_paid
    }

    return trades_df, summary


if __name__ == "__main__":
    be = compute_option_breakeven(spot=25145.0, vix=15.5, days_to_expiry=3.0, calendar_days_held=1)
    print("Black-Scholes Break-Even Audit Preview:")
    for k, v in be.items():
        print(f"  {k}: {v}")
