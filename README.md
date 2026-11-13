# ⚡ Nifty Quant Lab

A modular, self-hosted quantitative research and monitoring system for **Nifty 50, Bank Nifty, and Indian Index Options**.

> **Important Notice:** Strictly a research, backtesting, and forward paper-monitoring system. **No real order placement or broker trade execution capabilities.**

---

## 🌟 Key Features

1. **Lightweight Data Warehouse:** Ingests Nifty, Bank Nifty, India VIX, S&P 500 futures, Crude Oil, and USD/INR via Yahoo Finance and stores into **DuckDB** and **Apache Parquet**.
2. **Quantitative Feature Engineering:** Calculates price momentum, day close range position, VIX dynamics, and overnight global sentiment without lookahead bias.
3. **Walk-Forward Out-of-Sample Validation:** Avoids traditional overfitted train/test splits; uses rolling out-of-sample forward walks with XGBoost and regularized Logistic Regression.
4. **Realistic Indian Friction Engine:** Explicitly accounts for:
   - **STT (Securities Transaction Tax)**: 0.1% on options sell side
   - **Exchange Turnover Charges & GST**
   - **Stamp Duty**
   - **Realistic Bid-Ask Slippage** (1.5% each on entry & exit)
   - **Overnight Theta Decay**
5. **Simulated High-Conviction Risk Rules:** Only triggers signals when model probability $\ge 65\%$ or $\le 35\%$; otherwise sits in 100% simulated cash ("NO TRADE").
6. **Interactive Streamlit Dashboard:** Complete visual interface for backtest equity curves, feature distributions, and daily morning paper scanner.
7. **Automated Telegram Alerts:** Dispatches daily simulated research alerts directly to your mobile phone.

---

## 🚀 Quickstart Guide

### 1. Installation
Activate your virtual environment and install requirements:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(Optional: Add your `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` if you want mobile alerts)*

### 3. Fetch Historical Market Data
Run the data pipeline to download 2 years of market data into DuckDB and Parquet:
```bash
python src/data_pipeline.py
```

### 4. Run the Streamlit Research Dashboard
Launch the interactive web UI:
```bash
streamlit run dashboard/app.py
```

---

## 📁 Architecture Overview

```text
nifty_quant_lab/
├── data/                         # Local DuckDB database and Parquet caches
├── src/
│   ├── data_pipeline.py          # Market data ingestion from Yahoo Finance
│   ├── feature_engine.py         # Quant feature matrix calculation
│   ├── model_engine.py           # Walk-forward cross-validation
│   ├── backtest_engine.py        # Post-cost options backtesting simulator
│   └── alert_service.py          # Daily paper research Telegram alert
├── dashboard/
│   └── app.py                    # Streamlit visual dashboard
├── .env.example                  # Environment secrets template
└── requirements.txt              # Python package dependencies
```
