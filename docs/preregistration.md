# Model & Strategy Pre-Registration Protocol

**Registration Date**: 2026-11-14  
**Registration Commit Hash**: `2544a91464a12108595eddf555836f842d588979`  
**Protocol Status**: FROZEN — Zero Further Historical Tuning  
**Repository**: `nifty-quant-lab`

---

## 1. Objective & Purpose
To eliminate data snooping, p-hacking, and backtest overfitting, this document formally pre-registers the **four candidate quant models and their executable trading strategies** for live forward-testing on Indian Equities (NIFTY 50). 

Following this registration, all future model evaluations will be conducted strictly out-of-sample on incoming forward-test sessions logged in `data/predictions_log.parquet`. Historical backtests on 2019–2026 data are permanently closed.

---

## 2. Frozen Candidate Set

### Candidate 1: Model A (15:15 IST EOD Close Model)
* **Execution Strategy**: **Strategy S1 (Overnight Carry)**
* **Execution Window**: 15:15 – 15:20 IST entry, exit at 09:15 – 09:20 IST open.
* **Feature Cutoff**: Strictly $\le$ 15:15 IST.
* **Feature Set (7 Features)**:
  1. `nifty_ret_1d`: 1-day close-to-close return (decimal).
  2. `nifty_ret_5d`: 5-day rolling return (decimal).
  3. `nifty_range_pos`: Intraday range position $(P_{15:15} - L) / (H - L)$.
  4. `nikkei_ret_same_day`: Completed same-day Nikkei 225 return (closes 11:30 AM IST).
  5. `hangseng_ret_same_day`: Mid-session Hang Seng return (checked at 13:30 IST).
  6. `vix_close`: India VIX spot tick at 15:15 IST.
  7. `days_to_expiry`: Calendar days to next Thursday weekly expiry (rolls to 7 if Thursday).
* **Architecture**: Random Forest Classifier (100 trees, max_depth=6) + Gradient Boosting Quantile Regressors (q10, q50, q90).
* **Decision Rule**:
  - Class 2 (Gap Up): $P(\text{Up}) \ge 0.45$ AND $Q_{50} > \text{Break-Even Call Hurdle}$ $\to$ Buy ATM Weekly Call.
  - Class 1 (Gap Down): $P(\text{Down}) \ge 0.35$ AND $|Q_{50}| > \text{Break-Even Put Hurdle}$ $\to$ Buy ATM Weekly Put.
  - Class 0 (Flat): Avoid carry (overnight theta danger).

---

### Candidate 2: S&P 500 Implied Gap Rule (Heuristic Baseline)
* **Execution Strategy**: **Strategy S2 (Post-Open Intraday Trade)**
* **Execution Window**: Enter ~09:20 IST (after 09:15 open price is established), exit at 15:15 IST.
* **Feature Cutoff**: Finalized US session close (~01:30 IST). (Note: Strictly invalid for S1 overnight carry).
* **Heuristic Formula**:
  $$\text{Implied Gap \%} = 0.40 \times \text{sp500\_overnight\_ret}$$
* **Decision Rule**:
  - If $\text{Implied Gap} \ge +0.25\% \to$ Bias Gap-Up (Long ATM Call / Long Futures at 09:20).
  - If $\text{Implied Gap} \le -0.25\% \to$ Bias Gap-Down (Long ATM Put / Short Futures at 09:20).
  - Otherwise $\to$ Flat / No Trade.

---

### Candidate 3: Simplified Macro-3 Model (S&P + Crude + VIX)
* **Execution Strategy**: **Strategy S2 (Post-Open Intraday Trade)**
* **Execution Window**: Enter ~09:20 IST, exit at 15:15 IST.
* **Feature Set (3 Features)**:
  1. `sp500_overnight_ret`: Finalized S&P 500 close return from prior US session.
  2. `crude_overnight_ret`: NYMEX WTI Crude prompt month return.
  3. `vix_close`: Prior close India VIX.
* **Preprocessing**: `StandardScaler` fitted strictly on historical training fold.
* **Architecture**: Multiclass Logistic Regression ($C=1.0$, L2 penalty).
* **Empirical Walk-Forward Baseline**: Accuracy 56.62%, ROC-AUC 0.7201.

---

### Candidate 4: Live GIFT Nifty Implied Gap Rule
* **Execution Strategy**: **Strategy S2 (Post-Open Intraday Trade)**
* **Execution Window**: Enter ~09:20 IST, exit at 15:15 IST.
* **Input**: Live GIFT Nifty contract quote from official NSE IX feed at 08:45–09:10 IST.
* **Formula**:
  $$\text{GIFT Basis \%} = \frac{\text{GIFT Quote} - \text{Nifty Spot Prev Close}}{\text{Nifty Spot Prev Close}} \times 100.0$$
* **Decision Rule**:
  - If $\text{GIFT Basis} \ge +0.30\% \to$ Enter Bullish Intraday Call at 09:20 IST.
  - If $\text{GIFT Basis} \le -0.30\% \to$ Enter Bearish Intraday Put at 09:20 IST.
  - Otherwise $\to$ Neutral / No Trade.
* **Historical Data Caveat**: Historical GIFT ticks prior to live deployment are synthetic proxies ($0.45 \times \text{SP500}$). This rule is strictly forward-tested on live feed ticks.

---

## 3. Multiple Testing & Statistical Criteria

1. **Family-Wise Error Rate**:
   Evaluating 4 candidate models against the null hypothesis requires Bonferroni correction:
   $$\alpha_{\text{adjusted}} = \frac{0.05}{4} = \mathbf{0.0125}$$
   Across all 6 historical exploratory configurations, the threshold was $\alpha = 0.05 / 6 = \mathbf{0.00833}$.
2. **Success Criterion for Forward-Test**:
   A candidate will be considered to demonstrate a true statistical edge if and only if:
   - Paired out-of-sample forward-test accuracy delta vs S&P baseline achieves $p < 0.0125$.
   - Out-of-sample option P&L net of realistic friction (charges + 1 pt slippage) is positive with 95% bootstrap CI lower bound $> 0$.
   - Tested on $\ge 100$ independent forward-test sessions logged under `run_type = 'SCHEDULED'`.

---

## 4. Execution Policy & Governance
- **No In-Sample Tweaks**: Model feature lists, weights, and thresholds are locked as of this commit.
- **Run Type Segregation**: Only rows tagged `run_type = 'SCHEDULED'` on genuine NSE trading days are eligible for performance scoring. `MANUAL` and `TEST` executions are logged for debugging but quarantined from performance records.

