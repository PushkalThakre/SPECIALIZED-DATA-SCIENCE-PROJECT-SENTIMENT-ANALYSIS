# Technical Documentation — Month 5 Advanced Data Science Project

## Project Overview
This project implements two production-ready ML pipelines:
1. **Customer Churn Classification** (Deep Learning / Gradient Boosting)
2. **Supermarket Sales Forecasting** (Bidirectional LSTM / ARIMA)

Both models are served through a unified FastAPI application with monitoring and Docker support.

---

## Architecture

```
Raw Data → Preprocessor → Model Training → Saved Model
                                               ↓
User Request → FastAPI → Predictor (loads model) → Response
                ↓
          MetricsTracker → monitoring/metrics.json
```

---

## Models

### Churn Classification
- **Input**: 7 features (Tenure, MonthlyCharges, TotalCharges, Contract, PaymentMethod, PaperlessBilling, SeniorCitizen)
- **Architecture**: Dense(128) → BN → Dropout → Dense(64) → BN → Dropout → Dense(32) → Dense(1, sigmoid)
- **Loss**: Binary Crossentropy | **Optimizer**: Adam
- **Fallback**: GradientBoostingClassifier (if TensorFlow unavailable)

### Sales Forecasting
- **Input**: 7-day sliding window of scaled daily sales
- **Architecture**: BiLSTM(64) → Dropout → BiLSTM(32) → Dropout → Dense(16) → Dense(1)
- **Loss**: MSE | **Optimizer**: Adam
- **Fallback**: ARIMA(5,1,0) (if TensorFlow unavailable)

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/churn/predict` | Single customer churn prediction |
| POST | `/churn/batch_predict` | Batch churn prediction |
| GET | `/churn/metrics` | Churn model metrics |
| POST | `/sales/forecast` | Sales forecast for N days |
| GET | `/sales/metrics` | Sales model metrics |
| GET | `/metrics` | API usage metrics |

---

## How to Train

```bash
# Option 1: Train all models
python -m src.training.train_all

# Option 2: Train individually
python -m src.training.train_churn
python -m src.training.train_sales
```

## How to Serve

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
# Docs: http://localhost:8000/docs
```

## Docker

```bash
docker-compose up --build
```

---

## Technical Requirements Mapping

| Requirement | Implementation |
|-------------|---------------|
| Deep learning model | TF/Keras Dense + BiLSTM networks |
| Data preprocessing pipeline | ChurnPreprocessor + SalesPreprocessor classes |
| Containerized deployment | docker/Dockerfile + docker-compose.yml |
| Web API for model serving | FastAPI (src/api/app.py) |
| Model monitoring | MetricsTracker (src/monitoring/tracker.py) |
| Error handling & logging | try/except + loguru throughout |
| Testing | pytest in tests/ directory |

---

## Ethical Considerations
- Churn model trained on anonymised customer IDs only
- Predictions are probabilistic — human review recommended for high-stakes decisions
- Model drift monitoring in place to flag when retraining is needed
- No personally identifiable information stored in model weights
