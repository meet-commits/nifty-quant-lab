"""
src/backtest_engine.py
Purpose: Realistic Options Backtesting Engine with full Indian transaction friction:
- STT (Securities Transaction Tax)
- Exchange Turnover Charges (NSE)
- GST (18%)
- Stamp Duty
- Bid-Ask Spread Slippage
- Overnight Options Theta Decay
- Strict High-Conviction Probability Filtering
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

def run_realistic_backtest(
    predictions_df: pd.DataFrame,
    initial_capital: float = 100000.0,
    prob_threshold_high: float = 0.65,
    prob_threshold_low: float = 0.35,
    lot_size: int = 25,
    base_atm_premium: float = 160.0,
    overnight_theta_decay: float = 7.5,
    slippage_pct: float = 0.015,
    simulated_sl_pct: float = 0.25
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Simulates options trading on out-of-sample predictions.
    Strictly paper research only.
    """
    capital = initial_capital
    peak_capital = initial_capital
    max_drawdown_pct = 0.0

    trade_records = []

    for _, row in predictions_df.iterrows():
        prob = row['predicted_prob_up']
        gap_pct = row['actual_gap_pct']
        nifty_ref = row['nifty_close']

        # Signal Engine: Filter out low conviction noise
        if prob >= prob_threshold_high:
            signal = "CALL"
        elif prob <= prob_threshold_low:
            signal = "PUT"
        else:
            signal = "NO_TRADE"

        if signal == "NO_TRADE":
            continue

        # Points moved by the underlying index
        gap_points = (gap_pct / 100.0) * nifty_ref
        
        # Options Greeks Approximation (ATM Option Delta ~ 0.50)
        # Option Premium Change = (Index Points * Delta) - Overnight Theta Decay
        if signal == "CALL":
            delta_pnl_points = (gap_points * 0.50) - overnight_theta_decay
        else:
            delta_pnl_points = (-gap_points * 0.50) - overnight_theta_decay

        entry_premium = base_atm_premium
        # Apply simulated stop loss if gap moved violently against us
        min_allowed_premium = entry_premium * (1.0 - simulated_sl_pct)
        exit_premium = max(entry_premium + delta_pnl_points, min_allowed_premium)
        exit_premium = max(exit_premium, 2.0)  # Cannot be below penny value

        # ---------------- STATUTORY INDIAN TAXES & SLIPPAGE ----------------
        # 1. Slippage: 1.5% on Entry + 1.5% on Exit (Bid-Ask execution gap)
        slippage_cost = ((entry_premium * slippage_pct) + (exit_premium * slippage_pct)) * lot_size

        # 2. Indian Statutory Taxes
        buy_turnover = entry_premium * lot_size
        sell_turnover = exit_premium * lot_size

        # STT: 0.1% on Option Sell Turnover
        stt = sell_turnover * 0.0010
        # NSE Exchange Turnover: ~0.05% on total turnover
        exchange_turnover = (buy_turnover + sell_turnover) * 0.0005
        # GST: 18% on exchange charge
        gst = exchange_turnover * 0.18
        # Stamp duty: 0.003% on buy turnover
        stamp_duty = buy_turnover * 0.00003

        total_friction = slippage_cost + stt + exchange_turnover + gst + stamp_duty

        # P&L Calculation
        gross_pnl = (exit_premium - entry_premium) * lot_size
        net_pnl = gross_pnl - total_friction

        capital += net_pnl
        if capital > peak_capital:
            peak_capital = capital
        drawdown = (peak_capital - capital) / peak_capital * 100.0
        if drawdown > max_drawdown_pct:
            max_drawdown_pct = drawdown

        trade_records.append({
            "date": pd.to_datetime(row['date']).strftime('%Y-%m-%d'),
            "signal": signal,
            "probability": round(prob, 3),
            "gap_pct": round(gap_pct, 2),
            "entry_premium": round(entry_premium, 2),
            "exit_premium": round(exit_premium, 2),
            "gross_pnl": round(gross_pnl, 2),
            "friction_cost": round(total_friction, 2),
            "net_pnl": round(net_pnl, 2),
            "capital_after_trade": round(capital, 2),
            "drawdown_pct": round(drawdown, 2)
        })

    trades_df = pd.DataFrame(trade_records)

    # Compute Summary Statistics
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
