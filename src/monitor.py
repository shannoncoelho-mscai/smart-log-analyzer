"""
Smart Log Analyzer - Model Monitoring & Data Drift Detection
Authors: Lisa Luis, Shannon Coelho
MSC AI, Goa University

Monitors:
  1. Prediction distribution over time (concept drift)
  2. Feature distribution shift  (data drift)
  3. Accuracy on labelled production data
  4. Alerts when retraining is needed
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime

MONITOR_LOG = "models/monitoring_log.json"
DRIFT_THRESHOLD = 0.15   # PSI above this → alert
PERF_THRESHOLD  = 0.80   # Accuracy below this → alert


# ─────────────────────────────────────────────
# Population Stability Index  (PSI)
# ─────────────────────────────────────────────
def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    PSI < 0.1  : No significant change
    PSI < 0.25 : Moderate change — monitor closely
    PSI >= 0.25: Significant shift — retrain
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    expected_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_counts   = np.histogram(actual,   bins=breakpoints)[0] / len(actual)

    # Avoid log(0)
    expected_counts = np.where(expected_counts == 0, 1e-6, expected_counts)
    actual_counts   = np.where(actual_counts   == 0, 1e-6, actual_counts)

    psi = np.sum((actual_counts - expected_counts) * np.log(actual_counts / expected_counts))
    return float(round(psi, 4))


# ─────────────────────────────────────────────
# Main monitoring function
# ─────────────────────────────────────────────
def run_monitoring(
    train_df:   pd.DataFrame,
    prod_df:    pd.DataFrame,
    features:   list,
    y_true:     np.ndarray = None,
    y_pred:     np.ndarray = None,
):
    """
    Parameters
    ----------
    train_df : reference (training) feature DataFrame
    prod_df  : production feature DataFrame
    features : list of feature column names
    y_true   : ground-truth labels for production batch (optional)
    y_pred   : model predictions for production batch (optional)
    """
    timestamp = datetime.utcnow().isoformat()
    report    = {"timestamp": timestamp, "drift": {}, "performance": {}, "alerts": []}

    print(f"\n{'═'*55}")
    print(f"  MONITORING REPORT  —  {timestamp}")
    print(f"{'═'*55}")

    # ── 1. Data Drift (PSI per feature) ──────────────
    print("\n[1] Feature Drift (PSI):")
    for feat in features:
        if feat not in train_df.columns or feat not in prod_df.columns:
            continue
        psi = compute_psi(train_df[feat].values, prod_df[feat].values)
        report["drift"][feat] = psi

        status = "✅ OK"
        if psi >= 0.25:
            status = "🚨 RETRAIN"
            report["alerts"].append(f"HIGH DRIFT on '{feat}' (PSI={psi})")
        elif psi >= 0.1:
            status = "⚠️  MONITOR"

        print(f"  {feat:35s}  PSI={psi:.4f}  {status}")

    # ── 2. Prediction Performance ─────────────────────
    if y_true is not None and y_pred is not None:
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, zero_division=0)
        report["performance"] = {"accuracy": round(acc, 4), "f1": round(f1, 4)}

        print(f"\n[2] Production Performance:")
        print(f"  Accuracy : {acc:.4f}  {'🚨 BELOW THRESHOLD' if acc < PERF_THRESHOLD else '✅'}")
        print(f"  F1 Score : {f1:.4f}")

        if acc < PERF_THRESHOLD:
            report["alerts"].append(f"LOW ACCURACY: {acc:.4f} < threshold {PERF_THRESHOLD}")

    # ── 3. Prediction distribution ────────────────────
    if y_pred is not None:
        failure_rate = float(np.mean(y_pred))
        report["prediction_stats"] = {"failure_rate": round(failure_rate, 4)}
        print(f"\n[3] Prediction Stats:")
        print(f"  Failure rate in batch: {failure_rate:.2%}")

    # ── 4. Alerts summary ─────────────────────────────
    print(f"\n[ALERTS] {len(report['alerts'])} alert(s):")
    if report["alerts"]:
        for alert in report["alerts"]:
            print(f"  ⚠  {alert}")
    else:
        print("  None — system looks healthy.")

    # ── 5. Persist log ────────────────────────────────
    logs = []
    if os.path.exists(MONITOR_LOG):
        with open(MONITOR_LOG) as f:
            logs = json.load(f)
    logs.append(report)
    os.makedirs("models", exist_ok=True)
    with open(MONITOR_LOG, "w") as f:
        json.dump(logs, f, indent=2)

    print(f"\n[SAVED] Log appended → {MONITOR_LOG}")
    return report


# ─────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Synthetic demo: simulate a reference vs production batch
    np.random.seed(0)
    features = ["Air temperature [K]", "Torque [Nm]", "Rotational speed [rpm]"]

    train_data = {f: np.random.normal(300, 10, 1000) for f in features}
    prod_data  = {f: np.random.normal(305, 12, 200)  for f in features}   # slight drift

    train_df = pd.DataFrame(train_data)
    prod_df  = pd.DataFrame(prod_data)

    y_true = np.random.randint(0, 2, 200)
    y_pred = np.random.randint(0, 2, 200)

    run_monitoring(train_df, prod_df, features, y_true, y_pred)