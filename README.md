# 🔍 Smart Log Analyzer
### System Failure Prediction using Machine Learning & MLOps
**Authors:** Lisa Luis, Shannon Coelho  
**Program:** MSC AI, Goa University  
**Year:** 2026

---

##  Project Structure

```
smart_log_analyzer/
│
├── data/
│   └── ai4i2020.csv          ← Download from Kaggle (link below)
│
├── src/
│   ├── preprocess.py         ← Data cleaning, feature engineering, scaling
│   ├── train.py              ← Train 4 models, MLflow tracking, save best
│   └── monitor.py            ← Data drift (PSI) + performance monitoring
│
├── api/
│   └── app.py                ← Flask REST API for predictions
│
├── models/                   ← Auto-created after training
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── model_meta.json
│
├── streamlit_app.py          ← Interactive web dashboard
├── requirements.txt
└── README.md
```

---

##  Step-by-Step Setup

### Step 1 — Clone & Install

```bash
git clone https://github.com/<your-repo>/smart_log_analyzer.git
cd smart_log_analyzer
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Download Dataset

1. Go to: https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020
2. Download `ai4i2020.csv`
3. Place it in the `data/` folder

### Step 3 — Preprocess Data

```bash
python src/preprocess.py
```
This will:
- Load and clean the AI4I dataset
- Engineer 3 new features (Temp Delta, Power Proxy, Wear Stress)
- Split into train/test (80/20)
- Fit and save a `StandardScaler`

### Step 4 — Train Models

```bash
python src/train.py
```
This will train:
- Logistic Regression
- Decision Tree
- ✅ Random Forest (recommended)
- SVM

And will:
- Balance classes with SMOTE
- Track experiments with MLflow
- Save the best model by F1-score

To view the MLflow dashboard:
```bash
mlflow ui
# Open http://localhost:5000 in browser
```

### Step 5 — Run Flask API

```bash
python api/app.py
```

Test it:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551,
    "torque": 42.8,
    "tool_wear": 0
  }'
```

Expected response:
```json
{
  "prediction": 0,
  "label": "NORMAL",
  "confidence": 97.5,
  "failure_probability": 2.5,
  "risk_level": "LOW"
}
```

### Step 6 — Launch Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501 — you'll see:
- Live sliders for all sensor inputs
- Real-time failure prediction gauge
- Feature radar chart
- Risk level alerts

### Step 7 — Run Monitoring

```bash
python src/monitor.py
```

This runs drift detection using **Population Stability Index (PSI)** and logs results to `models/monitoring_log.json`.

---

##  API Endpoints

| Method | Endpoint       | Description                  |
|--------|---------------|------------------------------|
| POST   | `/predict`     | Predict failure from sensors |
| GET    | `/health`      | Health check                 |
| GET    | `/model-info`  | Model metadata + metrics     |

---

##  ML Pipeline

```
Raw CSV Data
     │
     ▼
Data Cleaning (handle missing, drop IDs)
     │
     ▼
Feature Engineering (+3 derived features)
     │
     ▼
Train/Test Split (80/20, stratified)
     │
     ▼
SMOTE Oversampling (balance failure class)
     │
     ▼
StandardScaler (fit on train only)
     │
     ▼
Train 4 Models → MLflow tracking
     │
     ▼
Select Best by F1 Score
     │
     ▼
Save model.pkl + scaler.pkl + meta.json
     │
     ▼
Flask API  ─────►  Streamlit Dashboard
     │
     ▼
Monitoring (PSI drift + accuracy check)
```

---

##  Expected Results (Random Forest)

| Metric    | Expected Value |
|-----------|---------------|
| Accuracy  | ~97–98%        |
| Precision | ~90–95%        |
| Recall    | ~85–92%        |
| F1 Score  | ~88–93%        |

---

##  Tools Used

| Tool          | Purpose                    |
|--------------|----------------------------|
| Python 3.11   | Core language              |
| Scikit-learn  | ML models                  |
| Pandas/NumPy  | Data processing            |
| imbalanced-learn | SMOTE oversampling      |
| MLflow        | Experiment tracking        |
| Flask         | REST API                   |
| Streamlit     | Web dashboard              |
| Plotly        | Interactive charts         |
| Git           | Version control            |

---

## Authors

- **Lisa Luis** — MSC AI, Goa University
- **Shannon Coelho** — MSC AI, Goa University

Submitted for SEA Project — February 2026