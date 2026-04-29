"""
Smart Log Analyzer - Data Preprocessing
Authors: Lisa Luis, Shannon Coelho
MSC AI, Goa University
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH   = "data/ai4i2020.csv"
OUTPUT_DIR  = "models"
TEST_SIZE   = 0.2
RANDOM_SEED = 42

FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET_COL = "Machine failure"


def load_data(path: str) -> pd.DataFrame:
    """Load CSV dataset and perform basic checks."""
    df = pd.read_csv(path)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"[INFO] Missing values:\n{df.isnull().sum()}\n")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnecessary columns, handle missing values."""
    # Drop non-numeric identifier columns if present
    drop_cols = [c for c in ["UDI", "Product ID"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Fill remaining numeric NaNs with column median
    for col in df.select_dtypes(include=[np.number]).columns:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    print(f"[INFO] After cleaning: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features that help predict failures."""
    df = df.copy()
    # Temperature delta — high delta often precedes failure
    df["Temp_Delta"] = df["Process temperature [K]"] - df["Air temperature [K]"]

    # Power proxy: torque × rotational speed (in radians/s)
    df["Power_Proxy"] = df["Torque [Nm]"] * (df["Rotational speed [rpm]"] * 2 * np.pi / 60)

    # Tool-wear × torque stress index
    df["Wear_Stress"] = df["Tool wear [min]"] * df["Torque [Nm]"]

    print("[INFO] Engineered features: Temp_Delta, Power_Proxy, Wear_Stress")
    return df


def split_and_scale(df: pd.DataFrame):
    """
    Split into train/test, scale features, persist scaler.

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    feature_names                      : list[str]
    """
    feature_cols = FEATURE_COLS + ["Temp_Delta", "Power_Proxy", "Wear_Stress"]
    # Keep only cols that exist in df
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    print(f"[INFO] Scaler saved → {OUTPUT_DIR}/scaler.pkl")
    print(f"[INFO] Train size: {X_train.shape}, Test size: {X_test.shape}")

    return X_train, X_test, y_train, y_test, feature_cols


# ─────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, features = split_and_scale(df)
    print("\n[DONE] Preprocessing complete.")
    print(f"       Features used: {features}")