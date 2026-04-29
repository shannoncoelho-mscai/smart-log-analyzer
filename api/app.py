"""
Smart Log Analyzer - Flask REST API
Authors: Lisa Luis, Shannon Coelho
MSC AI, Goa University

Endpoints:
  POST /predict      — predict failure from system parameters
  GET  /health       — health check
  GET  /model-info   — model metadata
"""

import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# ─────────────────────────────────────────────
# Load model artifacts on startup
# ─────────────────────────────────────────────
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH  = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
META_PATH   = os.path.join(MODEL_DIR, "model_meta.json")

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(META_PATH) as f:
    meta = json.load(f)

FEATURES = meta["features"]

app = Flask(__name__)
CORS(app)  # allow cross-origin requests from UI


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def build_feature_vector(data: dict) -> np.ndarray:
    """
    Construct the feature vector from raw input.
    Computes derived features (Temp_Delta, Power_Proxy, Wear_Stress).
    """
    air_temp   = float(data["air_temperature"])
    proc_temp  = float(data["process_temperature"])
    rpm        = float(data["rotational_speed"])
    torque     = float(data["torque"])
    tool_wear  = float(data["tool_wear"])

    temp_delta   = proc_temp - air_temp
    power_proxy  = torque * (rpm * 2 * 3.14159 / 60)
    wear_stress  = tool_wear * torque

    raw_map = {
        "Air temperature [K]":     air_temp,
        "Process temperature [K]": proc_temp,
        "Rotational speed [rpm]":  rpm,
        "Torque [Nm]":             torque,
        "Tool wear [min]":         tool_wear,
        "Temp_Delta":              temp_delta,
        "Power_Proxy":             power_proxy,
        "Wear_Stress":             wear_stress,
    }

    vec = np.array([raw_map[f] for f in FEATURES]).reshape(1, -1)
    return scaler.transform(vec)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": meta["model_name"]})


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_name": meta["model_name"],
        "features":   FEATURES,
        "metrics":    meta["metrics"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Request body (JSON):
    {
        "air_temperature":     298.1,
        "process_temperature": 308.6,
        "rotational_speed":    1551,
        "torque":              42.8,
        "tool_wear":           0
    }
    """
    try:
        data = request.get_json(force=True)

        required = ["air_temperature", "process_temperature",
                    "rotational_speed", "torque", "tool_wear"]
        missing = [k for k in required if k not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        X = build_feature_vector(data)

        pred   = int(model.predict(X)[0])
        proba  = model.predict_proba(X)[0].tolist() if hasattr(model, "predict_proba") else None
        label  = "FAILURE" if pred == 1 else "NORMAL"

        resp = {
            "prediction": pred,
            "label":      label,
            "confidence": round(max(proba) * 100, 2) if proba else None,
            "failure_probability": round(proba[1] * 100, 2) if proba else None,
        }

        # Risk level
        fp = resp["failure_probability"] or 0
        if fp >= 70:
            resp["risk_level"] = "CRITICAL"
        elif fp >= 40:
            resp["risk_level"] = "HIGH"
        elif fp >= 20:
            resp["risk_level"] = "MEDIUM"
        else:
            resp["risk_level"] = "LOW"

        return jsonify(resp)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[API] Model loaded: {meta['model_name']}")
    print(f"[API] Features: {FEATURES}")
    app.run(host="0.0.0.0", port=5000, debug=False)