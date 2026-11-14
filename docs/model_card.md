# Model Card: Dual-Horizon Overnight Gap Prediction Engine

## 1. Model Overview & Purpose

The **Dual-Horizon Overnight Gap Prediction Engine** is a quantitative machine learning and econometric system designed to estimate the directional probability and expected point magnitude of overnight index opening gaps for the **Nifty 50** and **Bank Nifty** on the National Stock Exchange of India (NSE).

- **Primary Goal**: Help quant researchers and systematic traders quantify overnight directional bias and compare expected gap magnitude against theoretical Black-Scholes option theta decay and statutory Indian market frictions.
- **System Architecture**:
  - **15:15 IST Close Model**: Fires at 15:15 IST (15 minutes prior to cash close) to support overnight carry decisions using strictly pre-close intermarket and intraday features.
  - **08:45–09:10 IST Pre-Open Nowcast Model**: Fires during the pre-open window, updating the distribution with overnight US market closes, morning GIFT Nifty prints, and overnight energy/macro moves.
- **Out of Scope**: High-frequency execution, intraday scalping, and automated broker order execution.

---

## 2. Target Variable Definition

The system models two complementary targets:

### A. 3-Class Directional Classification (`target_3class`)
To prevent the model from chasing noisy, sub-friction price movements, gaps are partitioned into three distinct classes:
$$\text{Gap \%}_{t+1} = \frac{\text{Open}_{t+1} - \text{Close}_t}{\text{Close}_t} \times 100$$

$$\text{Class} = \begin{cases}
+1 \ (\text{Gap Up}), & \text{if } \text{Gap \%}_{t+1} > +0.25\% \\
-1 \ (\text{Gap Down}), & \text{if } \text{Gap \%}_{t+1} < -0.25\% \\
0 \ (\text{Flat / Noise Band}), & \text{if } -0.25\% \le \text{Gap \%}_{t+1} \le +0.25\%
\end{cases}$$

- **Threshold Justification ($\pm 0.25\%$)**: For Nifty at ~24,000, $0.25\%$ represents ~60 index points. This hurdle is specifically calibrated to exceed typical overnight option bid-ask spread slippage (1.5%), round-trip statutory turnover charges, and 1-day theta decay.

### B. Quantile Gap Magnitude Regressor
To predict the distribution of gap magnitude in absolute index points ($\Delta_{\text{pts}} = \text{Open}_{t+1} - \text{Close}_t$), Gradient Boosting Quantile Regressors are fit at three loss quantiles:
- **$P_{10}$ (10th Percentile)**: Adverse downside tail estimate.
- **$P_{50}$ (Median)**: Expected base-case gap move.
- **$P_{90}$ (90th Percentile)**: Favorable upside tail estimate.

---

## 3. Feature Architecture & Information Timing

To strictly prohibit lookahead bias, features are partitioned strictly by timestamp availability.

### 15:15 IST Model (EOD Close Horizon)
All inputs are finalized by 15:15 IST on trading day $t$:
| Feature Category | Inputs & Transformations | Timestamp Verification |
| :--- | :--- | :--- |
| **Domestic Price Action** | `nifty_ret_1d`, `nifty_ret_5d`, `nifty_range_pct`, `close_loc_in_range` | Up to 15:15 IST intraday tick |
| **Implied Volatility** | `vix_level`, `vix_ret_5d`, `vix_above_ma5` (Ratio over dynamic 5-day SMA) | Up to 15:15 IST intraday tick |
| **Intermarket Benchmarks** | `es_intraday_ret` (US S&P futures move during Asia hours), `crude_ret_1d`, `usdinr_ret_1d` | Available live at 15:15 IST |
| **Asian Equity Closes** | Nikkei 225 return, Hang Seng return | Sessions close before 13:00 IST |
| **Calendar & Expiry** | `day_of_week` (Monday–Friday one-hot), `days_to_expiry` (weekly/monthly) | Deterministic calendar math |

### 08:45–09:10 IST Nowcast Model (Pre-Open Horizon)
In addition to all 15:15 IST features, the nowcast model incorporates overnight information:
- **US Cash Close**: Final close of S&P 500 (`^GSPC`) and NASDAQ from previous night.
- **Overnight Crude Oil**: Extended electronic session move of WTI Crude (`CL=F`).
- **GIFT Nifty Delta**: Morning trading level of GIFT Nifty relative to previous session cash close.

### Feature Engine Parity
The exact same function (`build_feature_matrix()` in `src/feature_engine.py`) is invoked for historical backtesting and live real-time inference (where the live snapshot is appended as row $T$). This guarantees zero train/serve skew and strictly prevents hardcoded heuristics.

---

## 4. Evaluation Protocol & Probability Calibration

### Walk-Forward Out-of-Sample Methodology
- **Expanding Window Architecture**: Training starts on a minimum seed window (~500 sessions) and steps forward in chunks of 60 trading days (~1 quarter).
- **No In-Sample Contamination**: For each chunk, the model trains strictly on data up to $t-1$. All reported performance metrics are purely out-of-sample forward projections.
- **Model Selection**: Calibrated regularized Logistic Regression ($L_2$) and shallow XGBoost / Gradient Boosting are evaluated. Simpler models are preferred unless complex trees demonstrate statistically significant outperformance.

### Calibration
Raw multiclass logits are calibrated via **Platt Scaling** (`CalibratedClassifierCV(method='sigmoid', cv=3)`) fit over the training fold. This ensures that a predicted probability of 60% corresponds to a true historical frequency of 60%.

### Baseline Comparisons
The model's performance must beat the following standard baselines out-of-sample:
1. **Majority Class Baseline**: Predicts the unconditional prior class distribution.
2. **Prior Day Gap Sign**: Predicts gap continuation from the prior day.
3. **S&P 500 Sign Baseline**: Correlates direction solely with the overnight US close.
4. **Brier Score Prior Baseline**: Evaluated against $p(1-p)$ prior variance; model Brier score must show positive information skill ($Brier_{\text{model}} < Brier_{\text{prior}}$).
5. **Bootstrap 95% Confidence Intervals**: Metrics are reported with bootstrap resampled empirical confidence intervals.

---

## 5. Black-Scholes Options Hurdle Calculation

Rather than asserting arbitrary risk-reward rules, signals are evaluated against theoretical Black-Scholes option pricing:

$$\Delta_{\text{Theta}} = -\frac{S \times \sigma \times \sqrt{\Delta t}}{\sqrt{2\pi}}$$

Where:
- $S$: Spot index price.
- $\sigma$: Implied Volatility derived from the live **India VIX** (`INDIAVIX / 100`).
- $\Delta t$: Exact calendar day fraction to next session ($\frac{1}{365}$ for weekday overnight carry, $\frac{3}{365}$ for Friday-to-Monday weekend carry).
- **Statutory Frictions**: Sourced dynamically from `config/charges.yaml`:
  - Securities Transaction Tax (STT): 0.1% on options sell turnover.
  - Exchange turnover fees, clearing charges, GST (18%), and stamp duty.
  - Realistic bid-ask spread slippage: 1.5% entry and exit.

A directional trade is labeled positive expectancy only if:
$$\mathbb{E}[\text{Payoff}] = (P_{\text{direction}} \times \text{Expected Gap Pts}) - (\text{Theta Decay}_{\Delta t} + \text{Round-trip Frictions}) > 0$$

---

## 6. Known Limitations & Failure Modes

1. **Overnight Black Swan Events**: Low-probability geopolitical events, unscheduled global policy announcements, or emergency central bank interventions occurring after 15:15 IST cannot be predicted by pre-close features.
2. **Volatility Crush (IV Collapse)**: Following major anticipated binary events (Union Budget, general election results, RBI monetary policy), an index gap in the predicted direction may still yield net negative option P&L due to sharp post-open IV compression.
3. **Stale Data Feeds**: `yfinance` endpoints are delayed up to 15 minutes. In high-volatility close auctions (15:20–15:30 IST), closing prices may diverge from the 15:15 snapshot.
4. **Regime Shifts**: Structural changes in index constituent weightings, liquidity conditions, or global macroeconomic correlation can reduce model predictive power until the walk-forward retraining window incorporates the new regime.

