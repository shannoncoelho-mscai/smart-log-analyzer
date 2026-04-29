import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

with open(os.path.join(MODEL_DIR, "model_meta.json")) as f:
    meta = json.load(f)

FEATURES = meta["features"]

app = Flask(__name__)
CORS(app)


def build_feature_vector(data):
    air = float(data["air_temperature"])
    proc = float(data["process_temperature"])
    rpm = float(data["rotational_speed"])
    torque = float(data["torque"])
    wear = float(data["tool_wear"])

    vec_map = {
        "Air temperature [K]": air,
        "Process temperature [K]": proc,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear,
        "Temp_Delta": proc - air,
        "Power_Proxy": torque * (rpm * 2 * 3.14159 / 60),
        "Wear_Stress": wear * torque,
    }

    X = np.array([vec_map[f] for f in FEATURES]).reshape(1, -1)
    return scaler.transform(X)


@app.route("/")
def home():
    return {"message": "Smart Log Analyzer API running"}


@app.route("/health")
def health():
    return {"status": "ok"}


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        X = build_feature_vector(data)
        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]

        return {
            "prediction": pred,
            "label": "FAILURE" if pred else "NORMAL",
            "confidence": float(max(proba)) * 100,
            "failure_probability": float(proba[1]) * 100
        }

    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    app.run()