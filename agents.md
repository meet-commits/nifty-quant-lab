# AGENTS.md — APEX QUANT 999 (educational Nifty/BankNifty/Sensex research app)

## Stack
Python, Streamlit, DuckDB, yfinance, XGBoost/sklearn, NSE MCP (mcp.nseindia.in). App entry: dashboard/app.py, modules in src/.

## Non-negotiable rules
1. NEVER fabricate market data, prices, OI, PCR, news, headlines or sources. If a feed fails, show "UNAVAILABLE" / last-known value with its timestamp. No synthetic candles, no hardcoded fallback prices shown as live.
2. Every market datapoint carries {value, timestamp (IST), source, is_stale}. UI badges must reflect real freshness, not "got a number".
3. All times use Asia/Kolkata via zoneinfo. Never naive datetime.now().
4. No look-ahead: a feature may only use data available at the stated prediction time. Training and live inference must share ONE feature-building code path.
5. Model metrics must be out-of-sample, expanding-window walk-forward, with baselines and confidence intervals. Never tune thresholds on the test set.
6. No trading-advice wording in UI ("stop loss", "buy zone", "carry CE/PE"). Use neutral terms: "reference level", "model signal". Keep disclaimers.
7. Secrets only via st.secrets/env. Escape all external text with html.escape before st.html.
8. Statutory rates (STT, GST, charges) live in config/charges.yaml, never hardcoded in logic.
9. Don't touch unrelated files. Before any large refactor or deletion, show a plan and wait for approval.
10. After each phase: run tests (pytest), run `python -m compileall src dashboard`, and summarize what changed + what is still unverified.