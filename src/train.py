"""
Smart Log Analyzer - Model Training
Authors: Lisa Luis, Shannon Coelho
MSC AI, Goa University

Trains Logistic Regression, Decision Tree, Random Forest, and SVM.
Logs metrics with MLflow. Saves best model.
"""

import os
import sys
import json
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     classification_report)
from imblearn.over_sampling  import SMOTE

# ─────────────────────────────────────────────
# 🔥 FIX: Force MLflow to use local safe folder
# ─────────────────────────────────────────────
mlflow.set_tracking_uri("file:./mlruns")

# Add src/ to path so we can import preprocess
sys.path.append(os.path.dirname(__file__))
from preprocess import load_data, clean_data, engineer_features, split_and_scale

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH  = "data/ai4i2020.csv"
OUTPUT_DIR = "models"
EXPERIMENT = "SmartLogAnalyzer"

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=10,
                                                  random_state=42, n_jobs=-1),
    "SVM":                 SVC(kernel="rbf", probability=True, random_state=42),
}


def evaluate(model, X_test, y_test, model_name: str) -> dict:
    """Compute and print classification metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
    }

    print(f"\n{'─'*50}")
    print(f"  {model_name}")
    print(f"{'─'*50}")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print(classification_report(y_test, y_pred,
                                 target_names=["Normal", "Failure"],
                                 zero_division=0))
    return metrics


def train_all(X_train, y_train, X_test, y_test):
    """Train all models, track with MLflow, return best model + metrics."""

    # Balance classes
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    print(f"[INFO] After SMOTE — Normal: {(y_res==0).sum()}, Failure: {(y_res==1).sum()}")

    mlflow.set_experiment(EXPERIMENT)

    best_f1 = 0.0
    best_model = None
    best_name = ""
    all_metrics = {}

    for name, model in MODELS.items():

        with mlflow.start_run(run_name=name):

            print(f"\n[TRAINING] {name} ...")

            model.fit(X_res, y_res)

            metrics = evaluate(model, X_test, y_test, name)

            # ✅ Log everything
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            all_metrics[name] = metrics

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_model = model
                best_name = name

    return best_model, best_name, all_metrics


def save_artifacts(model, model_name: str, metrics: dict, features: list):
    """Save best model and metadata"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_path = os.path.join(OUTPUT_DIR, "best_model.pkl")
    joblib.dump(model, model_path)

    print(f"\n[SAVED] Best model ({model_name}) → {model_path}")

    meta = {
        "model_name": model_name,
        "metrics": metrics,
        "features": features,
    }

    with open(os.path.join(OUTPUT_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[SAVED] Metadata → {OUTPUT_DIR}/model_meta.json")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Preprocess
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)

    X_train, X_test, y_train, y_test, features = split_and_scale(df)

    # 2. Train
    best_model, best_name, all_metrics = train_all(
        X_train, y_train, X_test, y_test
    )

    # 3. Save
    save_artifacts(best_model, best_name, all_metrics[best_name], features)

    print(f"\n✅ Best model: {best_name} | F1: {all_metrics[best_name]['f1']:.4f}")