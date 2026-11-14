# ⚡ Nifty Quant Lab

A modular, self-hosted quantitative research and monitoring system for **Nifty 50, Bank Nifty, and Indian Index Options**.

> **Important Notice:** Strictly a research, backtesting, and forward paper-monitoring system. **No real order placement or broker trade execution capabilities.**

---

## ⚠️ Data Sources and Limits

### 1. Yahoo Finance (`yfinance`)
- **Latency & Accuracy**: Data fetched via `yfinance` is typically **delayed by 15 minutes** during active market hours.
- **Unofficial Interface**: Relies on public upstream endpoints that can be subject to unannounced schema changes, rate limits, HTTP 429 throttling, or transient connectivity hiccups.
- **Usage Boundary**: Designed strictly for historical exploratory data analysis, offline feature engineering, and model training. Never rely on `yfinance` for low-latency live execution or critical intraday decision-making.

### 2. NSE MCP Live & Bhavcopy Server (`mcp.nseindia.in`)
- **Educational / Research Scope**: The Model Context Protocol (MCP) client connects to official NSE endpoints for research and educational purposes only.
- **Session Lifecycle & Rate Limits**: Requests are routed through session-managed HTTP pools with exponential backoff and jitter. NSE session tokens expire periodically and require renegotiation.
- **Resilient Fallback**: In the event of network disruption, IP rate limiting, or exchange maintenance windows, the system automatically degrades gracefully to cached local DuckDB historical snapshots without raising uncaught exceptions or interrupting UI rendering.

### 3. Local DuckDB & Apache Parquet Warehouse
- **Offline Persistence**: All daily bhavcopies and ~10 years of multi-asset OHLCV records are indexed into `data/research.duckdb` and `data/market_data.parquet`.
- **Idempotency**: All ingestion routines use transactional `ON CONFLICT DO NOTHING` / deduplicated primary keys (`(symbol, trade_date)`) to guarantee data integrity across repeated pipeline runs.

---

## 🌟 Key Features

1. **Lightweight Data Warehouse:** Ingests Nifty, Bank Nifty, India VIX, S&P 500 futures, Crude Oil, and USD/INR into **DuckDB** and **Apache Parquet**.
2. **Dual-Horizon Gap Prediction:**
   - **15:15 IST Close Model**: Strictly uses data available before market close (Nikkei/Hang Seng closes, intraday ES move, VIX, Crude, USD/INR, days-to-expiry, day-of-week). Zero future lookahead.
   - **08:45–09:10 IST Nowcast Model**: Integrates overnight US closes, GIFT Nifty opening prints, and European/Asian sentiment.
3. **Calibrated 3-Class & Quantile Targets:**
   - 3-Class Classification: **Gap Up (> +0.25%)**, **Gap Down (< -0.25%)**, and **Flat ([-0.25%, +0.25%])**.
   - Quantile Regressor: Predicts expected point move at 10th ($P_{10}$), 50th ($P_{50}$ median), and 90th ($P_{90}$) percentiles.
4. **Walk-Forward Out-of-Sample Validation:**
   - Expanding-window walk-forward training (retrained every 60 sessions).
   - Calibrated probabilities via Platt scaling / Isotonic regression with out-of-fold cross-validation.
   - Benchmark evaluation against Majority Class, S&P 500 Sign, Gap Momentum, and Prior Brier Score ($p(1-p)$).
5. **Black-Scholes Options Hurdle Engine:**
   - Replaces naive heuristic gap formulas with theoretical Black-Scholes pricing.
   - Computes overnight Theta decay over calendar days (including multi-day weekend carry) using live India VIX as IV.
   - Compares expected gap payoff against ATM premium decay + statutory friction (STT, GST, exchange charges from `config/charges.yaml`).
6. **Dynamic Bhavcopy Scanner & Whole-Universe Breadth:**
   - Scans 2,000+ NSE equities loaded into DuckDB via vectorized pandas indicator pipelines (EMA 20/50/200, Wilder's RSI-14, ATR-14, 20d Volume Surge).
   - Computes factual market breadth (% above key EMAs, Advance/Decline) from the entire universe with explicit `as_of_date`.
7. **Modular Streamlit Dashboard:** Cleanly partitioned components with `st.fragment` selective auto-refreshing during market hours and zero CPU burn off-hours.

---

## 🚀 Quickstart Guide

### 1. Installation
Activate your virtual environment and install pinned dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(Optional: Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` for daily research alerts)*

### 3. Fetch Historical Market Data & Bhavcopy
Ingest ~10 years of multi-asset history and recent bhavcopy universe:
```bash
python src/data_pipeline.py
python src/bhavcopy_service.py
```

### 4. Run Test Suite & Linting Gate
Verify mathematical integrity and absence of lookahead leakage:
```bash
ruff check .
pytest tests/ -v
```

### 5. Launch the Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📁 Architecture Overview

```text
nifty_quant_lab/
├── config/
│   ├── charges.yaml              # Statutory Indian exchange transaction taxes (STT, GST, stamp)
│   └── weights.csv               # Official index constituent weights with effective date
├── data/                         # Local DuckDB database and Parquet caches
│   ├── market_data.parquet       # Multi-asset OHLCV history (~10 years)
│   ├── predictions_log.parquet   # Forward-testing out-of-sample prediction audit log
│   └── research.duckdb           # DuckDB database with full bhavcopy_daily tables
├── src/
│   ├── data_pipeline.py          # Multi-asset history ingestion & caching
│   ├── feature_engine.py         # Leakage-free feature computation (15:15 IST & 08:45 IST)
│   ├── model_engine.py           # Walk-forward cross-validation, 3-class & quantile models
│   ├── backtest_engine.py        # Black-Scholes options hurdle & friction backtest engine
│   ├── bhavcopy_service.py       # NSE Bhavcopy ingestion, DuckDB caching & universe seeding
│   ├── chartink_engine.py        # Vectorized technical indicator queries & whole-universe breadth
│   ├── gap_prediction_service.py # Live dual-horizon inference service & audit logging
│   ├── live_market_service.py    # Resilient live snapshot with MCP & fallback providers
│   ├── macro_story_engine.py     # Factual 4-pillar market regime intelligence
│   └── nse_mcp_client.py         # Session-managed NSE MCP client
├── dashboard/
│   ├── app.py                    # Streamlit entry point (layout, fragment orchestrator)
│   └── components/               # Modular UI components (status bar, cockpit, scanner, etc.)
├── docs/
│   └── model_card.md             # Quantitative model card & validation specification
├── tests/                        # Full test suite (leakage, resilience, scanner, math)
├── pyproject.toml                # Pytest & Ruff configuration
└── requirements.txt              # Pinned Python package dependencies
```
