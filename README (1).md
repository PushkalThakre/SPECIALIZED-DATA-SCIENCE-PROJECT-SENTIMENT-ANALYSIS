# 🤖 Month 5 — Advanced Data Science Project
### Customer Churn Prediction + Sales Forecasting with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://tensorflow.org)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://docker.com)

---

## 📋 Project Overview

This project implements **two production-ready ML pipelines** using the provided datasets:

| Model | Dataset | Algorithm | Task |
|-------|---------|-----------|------|
| Churn Classifier | `customer_churn.csv` | Deep Neural Network | Binary Classification |
| Sales Forecaster | `supermarket_sales.csv` | Bidirectional LSTM | Time Series Forecasting |

Both models are served through a unified **FastAPI** REST API, containerized with **Docker**, and monitored in production.

---

## 🚀 Quick Start

### Option 1 — Local (Recommended for development)
```bash
# Install dependencies
pip install -r requirements.txt

# Run EDA
python notebooks/01_exploratory_analysis.py

# Train both models
python -m src.training.train_all

# Start API server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Option 2 — Docker
```bash
docker-compose up --build
```

### Run Tests
```bash
python -m pytest tests/ -v
```

### API Documentation
Once running: **http://localhost:8000/docs**

---

## 📁 Project Structure

```
month5-advanced-ds/
├── README.md                         ← You are here
├── requirements.txt                  ← Python dependencies
├── docker-compose.yml                ← Multi-container setup
├── Dockerfile → docker/Dockerfile    ← Container configuration
│
├── data/
│   ├── raw/
│   │   ├── customer_churn.csv        ← Churn dataset (500 rows)
│   │   └── supermarket_sales.csv     ← Sales dataset (2000 rows)
│   └── processed/                    ← Saved models + reports (auto-created)
│
├── notebooks/
│   └── 01_exploratory_analysis.py   ← EDA for both datasets
│
├── src/
│   ├── data_processing/
│   │   ├── churn_preprocessor.py    ← Encoding, scaling, validation
│   │   └── sales_preprocessor.py   ← Time series aggregation + sequences
│   ├── models/
│   │   ├── churn_model.py           ← DNN + sklearn fallback
│   │   └── sales_model.py          ← BiLSTM + ARIMA fallback
│   ├── training/
│   │   ├── train_churn.py           ← Churn training pipeline
│   │   ├── train_sales.py          ← Sales training pipeline
│   │   └── train_all.py            ← Master training script ⭐
│   ├── inference/
│   │   └── predictor.py            ← ChurnPredictor + SalesPredictor
│   ├── api/
│   │   └── app.py                  ← FastAPI endpoints
│   └── monitoring/
│       └── tracker.py              ← Metrics + drift detection
│
├── tests/
│   ├── test_preprocessors.py
│   ├── test_models.py
│   └── test_api.py
│
├── docker/
│   └── Dockerfile
├── deployment/
│   ├── kubernetes.yml
│   └── deploy.sh
├── monitoring/
│   ├── prometheus.yml
│   └── dashboard_config.json
├── docs/
│   └── technical_documentation.md
└── scripts/
    ├── run_pipeline.sh
    └── quick_test.py
```

---

## 🌐 API Endpoints

```
GET  /health                  → System health check
GET  /docs                    → Interactive Swagger UI

POST /churn/predict           → Single customer churn prediction
POST /churn/batch_predict     → Batch predictions
GET  /churn/metrics           → Model evaluation metrics

POST /sales/forecast          → Sales forecast for next N days
GET  /sales/metrics           → Forecasting model metrics

GET  /metrics                 → API monitoring (latency, errors, counts)
```

### Example — Churn Prediction
```bash
curl -X POST http://localhost:8000/churn/predict \
  -H "Content-Type: application/json" \
  -d '{"Tenure":12,"MonthlyCharges":65.5,"TotalCharges":786.0,
       "Contract":"Month-to-month","PaymentMethod":"Electronic Check",
       "PaperlessBilling":"Yes","SeniorCitizen":0}'
```

### Example — Sales Forecast
```bash
curl -X POST http://localhost:8000/sales/forecast \
  -H "Content-Type: application/json" \
  -d '{"recent_sales":[1200,1350,980,1100,1450,1300,1250],"forecast_days":7}'
```

---

## 📊 Expected Model Performance

| Model | Metric | Expected Value |
|-------|--------|---------------|
| Churn DNN | Accuracy | ~85–90% |
| Churn DNN | ROC-AUC | ~0.87–0.92 |
| Sales LSTM | MAE | ~50–120 |
| Sales LSTM | MAPE | ~8–15% |

*(Actual results vary; run train_all.py to see your results)*

---

## 🐳 Docker Details

```bash
# Build only
docker build -t ds-project-api -f docker/Dockerfile .

# Run with compose (API + Prometheus)
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop
docker-compose down
```

---

## 📚 Datasets Used
- **customer_churn.csv** — 500 customers, 9 features, binary churn label
- **supermarket_sales.csv** — 2000 transactions across 3 branches, used for daily sales aggregation

---

## ✅ Technical Requirements Checklist

- [x] Deep learning model (TF/Keras Dense + BiLSTM)
- [x] Complete preprocessing pipeline (encoding, scaling, sequence creation)
- [x] Containerized with Docker + docker-compose
- [x] REST API (FastAPI with Swagger docs)
- [x] Model monitoring (MetricsTracker + DriftDetector)
- [x] Error handling + loguru logging throughout
- [x] Test suite (pytest, 3 test files)
- [x] Production-ready code with documentation
