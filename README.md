# 💳 Real-Time Credit Card Fraud Detection System

A production-ready, end-to-end machine learning system for detecting fraudulent credit card transactions in real-time. Built as a comprehensive learning project covering all phases from data exploration to deployment and monitoring.

## 📋 Problem Statement

Credit card fraud costs the global economy over **$30 billion annually**. This project builds an ML system that:
- **Detects fraudulent transactions in real-time** (~5ms per prediction)
- **Minimizes business cost** by optimizing the precision-recall tradeoff
- **Explains predictions** to non-technical stakeholders using SHAP
- **Monitors for data drift** and model decay in production

### Business Cost Analysis
| Event | Cost | Impact |
|-------|------|--------|
| **False Negative** (missed fraud) | ~$500 | Bank absorbs the loss |
| **False Positive** (blocked legit) | ~$10 | Customer inconvenience, service call |
| **True Positive** (caught fraud) | $0 | Fraud prevented! |

> **Key Insight**: Missing a fraud is **50x more expensive** than a false alarm, so our model is tuned for high recall.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Training Pipeline                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │ Data     │→ │ Preproc  │→ │ Train &  │→ │ Save   │  │
│  │ Loading  │  │ & Scale  │  │ Evaluate │  │ Model  │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘  │
│       ↓              ↓             ↓            ↓       │
│    EDA Plots     Scaler.pkl   MLflow Logs   model.joblib│
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  Serving Pipeline                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │ FastAPI  │→ │ Predict  │→ │ Log &    │→ │Response│  │
│  │ /predict │  │ Pipeline │  │ Monitor  │  │ JSON   │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘  │
│       ↑                            ↓                    │
│   Streamlit              Prediction Logs                │
│   Dashboard              + Drift Detection              │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Rows | 284,807 transactions |
| Features | 30 (Time, V1–V28 PCA-transformed, Amount) |
| Target | Class (0=Legit, 1=Fraud) |
| Fraud Rate | 0.17% (492 frauds) — **extreme imbalance** |

---

## 🧪 Results

| Model | Resampling | PR-AUC | Recall | Precision | F1 | ROC-AUC |
|-------|-----------|--------|--------|-----------|-----|---------|
| **XGBoost** | SMOTE | **~0.85** | **0.85** | **0.82** | **0.83** | **0.98** |
| LightGBM | SMOTE | ~0.83 | 0.83 | 0.80 | 0.81 | 0.97 |
| Random Forest | SMOTE | ~0.80 | 0.80 | 0.78 | 0.79 | 0.97 |
| Logistic Regression | SMOTE | ~0.70 | 0.90 | 0.05 | 0.10 | 0.97 |

> **Note**: Actual values will vary. Run the pipeline to get your results.

> **Why not accuracy?** A model predicting "legit" for everything achieves 99.83% accuracy but catches **zero** frauds. We use PR-AUC as our primary metric.

---

## 📁 Project Structure

```
├── api/
│   ├── app.py              # FastAPI REST API with /predict and /health
│   └── schemas.py           # Pydantic input/output validation
├── scripts/
│   ├── train.py             # Main training pipeline orchestrator
│   ├── simulate_stream.py   # Real-time transaction streamer
│   └── run_dashboard.py     # Streamlit monitoring dashboard
├── src/
│   ├── config.py            # Central configuration (paths, seeds, costs)
│   ├── data_loader.py       # Data loading and validation
│   ├── eda.py               # Exploratory data analysis & plots
│   ├── preprocessing.py     # Feature scaling, train-test split
│   ├── resampling.py        # SMOTE, undersampling, class weights
│   ├── models.py            # sklearn Pipelines (LR, RF, XGB, LGBM)
│   ├── evaluation.py        # Metrics, PR/ROC curves, threshold tuning
│   ├── tuning.py            # RandomizedSearchCV & Optuna
│   ├── explainability.py    # SHAP feature importance & explanations
│   ├── experiment_tracking.py # MLflow logging wrapper
│   └── monitoring.py        # PSI drift detection, prediction logging
├── tests/
│   ├── test_preprocessing.py # Data pipeline unit tests
│   ├── test_api.py           # API endpoint tests
│   └── test_model.py         # Model training/prediction tests
├── models/                   # Saved model artifacts (joblib)
├── plots/                    # Generated EDA & evaluation plots
├── logs/                     # Prediction logs for monitoring
├── mlruns/                   # MLflow experiment tracking
├── creditcard.csv            # Dataset
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container build instructions
├── docker-compose.yml        # Container orchestration
├── .gitignore
└── README.md                 # You are here!
```

---

## 🚀 How to Run

### 1. Setup
```bash
# Clone the repository
git clone <repo-url>
cd credit-card-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Download the dataset from Kaggle and place creditcard.csv in the project root
```

### 2. Train the Models
```bash
# Full pipeline (EDA + Training + Evaluation + SHAP + MLflow)
python scripts/train.py

# Skip EDA (faster)
python scripts/train.py --skip-eda

# Include hyperparameter tuning (much slower, better results)
python scripts/train.py --tune

# Skip MLflow logging
python scripts/train.py --skip-mlflow
```

### 3. Start the API
```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test with curl
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0.0,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
    "Amount": 149.62
  }'
```

### 5. Simulate Real-Time Streaming
```bash
# Send 100 transactions to the API with random delays
python scripts/simulate_stream.py --n 100

# Custom API URL
python scripts/simulate_stream.py --n 50 --url http://localhost:8000
```

### 6. Launch Monitoring Dashboard
```bash
streamlit run scripts/run_dashboard.py
```

### Streamlit Community Cloud
Set the app's **Main file path** to `streamlit_app.py`. The FastAPI file
`api/app.py` serves the prediction API and is not the Streamlit dashboard.

### 7. Run Tests
```bash
pytest tests/ -v
```

### 8. Docker
```bash
# Build the container
docker build -t fraud-detection-api .

# Run it
docker run -p 8000:8000 fraud-detection-api

# Or use Docker Compose
docker-compose up --build
```

---

## 🔬 Key ML Decisions Explained

### Why PR-AUC > ROC-AUC?
ROC-AUC includes True Negatives, which dominate (99.83%). This inflates the metric. PR-AUC focuses exclusively on the minority class (fraud).

### Why SMOTE > Random Undersampling?
Undersampling discards 99.8% of legitimate transactions, losing valuable information. SMOTE creates synthetic fraud examples via interpolation.

### Why sklearn Pipelines?
Pipelines bundle preprocessing + model into one object, preventing data leakage and ensuring the exact same transformations at training and inference.

### Why Business-Cost Threshold Tuning?
The default 0.5 threshold is arbitrary. We sweep thresholds to find the one that minimizes: `cost = FN × $500 + FP × $10`.

---

## 🔮 Future Improvements

- [ ] **Real Kafka streaming** instead of simulated REST calls
- [ ] **Model registry** (MLflow Model Registry) for staging/production promotion
- [ ] **A/B testing** framework to compare models in production
- [ ] **Feature store** (Feast) for real-time feature engineering
- [ ] **Kubernetes deployment** with autoscaling based on traffic
- [ ] **Alerting** (PagerDuty/Slack) when drift is detected
- [ ] **AutoML** integration (Auto-sklearn/FLAML) for model selection
- [ ] **Graph-based features** from transaction networks

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| ML | scikit-learn, XGBoost, LightGBM, imbalanced-learn |
| Explainability | SHAP |
| Tuning | Optuna, RandomizedSearchCV |
| Tracking | MLflow |
| API | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit |
| Testing | pytest, httpx |
| Containerization | Docker, Docker Compose |
| Visualization | matplotlib, seaborn |

---

## 📄 License

This project is for educational purposes.

---

*Built as a mentored AI/ML engineering project covering the full MLOps lifecycle.*
