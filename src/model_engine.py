"""
src/model_engine.py
Purpose: Walk-Forward Validation Engine using XGBoost and Logistic Regression.
Evaluates out-of-sample predictability and protects against overfitting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from feature_engine import build_feature_matrix, FEATURE_COLUMNS

def train_and_evaluate_walk_forward(
    df: pd.DataFrame = None,
    train_window: int = 250,
    test_step: int = 40,
    model_type: str = "xgboost"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes walk-forward cross-validation.
    Simulates reality: The model is trained ONLY on historical data up to Day T,
    then predicts out-of-sample Day T+1 to Day T+k.
    """
    if df is None:
        df = build_feature_matrix()

    total_rows = len(df)
    if total_rows < train_window + test_step:
        raise ValueError(f"Insufficient data rows ({total_rows}). Need at least {train_window + test_step}.")

    print(f"🚀 Running Walk-Forward Validation ({model_type.upper()})...")
    print(f"   Total Samples: {total_rows} days | Train Window: {train_window} | Step: {test_step}")

    predictions = []
    start_idx = 0

    while start_idx + train_window + test_step <= total_rows:
        train_df = df.iloc[start_idx : start_idx + train_window]
        test_df = df.iloc[start_idx + train_window : start_idx + train_window + test_step]

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df['target']
        X_test = test_df[FEATURE_COLUMNS]
        y_test = test_df['target']

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model Selection
        if model_type.lower() == "xgboost" and XGB_AVAILABLE:
            model = XGBClassifier(
                n_estimators=35,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss"
            )
            model.fit(X_train_scaled, y_train)
            prob_up = model.predict_proba(X_test_scaled)[:, 1]
        else:
            # Fallback / Baseline: Regularized Logistic Regression
            model = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', random_state=42)
            model.fit(X_train_scaled, y_train)
            prob_up = model.predict_proba(X_test_scaled)[:, 1]

        # Record out-of-sample predictions
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            predictions.append({
                "date": row['date'],
                "nifty_close": row['nifty_close'],
                "actual_gap_pct": row['gap_pct'],
                "actual_target": int(row['target']),
                "predicted_prob_up": float(prob_up[i])
            })

        start_idx += test_step

    res_df = pd.DataFrame(predictions)

    # Compute overall Out-of-Sample Metrics
    binary_preds = (res_df['predicted_prob_up'] >= 0.50).astype(int)
    acc = accuracy_score(res_df['actual_target'], binary_preds)
    try:
        roc = roc_auc_score(res_df['actual_target'], res_df['predicted_prob_up'])
    except Exception:
        roc = 0.50

    prec = precision_score(res_df['actual_target'], binary_preds, zero_division=0)

    metrics = {
        "total_test_sessions": len(res_df),
        "out_of_sample_accuracy": round(acc * 100, 2),
        "roc_auc_score": round(roc, 3),
        "precision": round(prec * 100, 2)
    }

    print("\n--- Out-of-Sample Performance Report ---")
    print(f"Total Test Days: {metrics['total_test_sessions']}")
    print(f"Accuracy: {metrics['out_of_sample_accuracy']}%")
    print(f"ROC-AUC: {metrics['roc_auc_score']} (0.50 is purely random)")
    print(f"Precision: {metrics['precision']}%")

    return res_df, metrics

if __name__ == "__main__":
    res_df, metrics = train_and_evaluate_walk_forward()
