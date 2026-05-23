"""
app.py
FastAPI application exposing prediction endpoints for both models.
Run: uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import os
from loguru import logger

from src.monitoring.tracker import MetricsTracker

app = FastAPI(
    title="Advanced DS Project — Prediction API",
    description="Customer Churn Classification & Sales Forecasting API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

tracker = MetricsTracker()

# ── Request / Response Schemas ────────────────────────────────────────────────

class ChurnRequest(BaseModel):
    Tenure: float = Field(..., example=12, description="Customer tenure in months")
    MonthlyCharges: float = Field(..., example=65.5, description="Monthly charges in currency")
    TotalCharges: float = Field(..., example=786.0, description="Total charges to date")
    Contract: str = Field(..., example="Month-to-month", description="Contract type")
    PaymentMethod: str = Field(..., example="Electronic Check", description="Payment method")
    PaperlessBilling: str = Field(..., example="Yes", description="Paperless billing")
    SeniorCitizen: int = Field(..., example=0, description="Senior citizen flag (0/1)")

class ChurnBatchRequest(BaseModel):
    customers: List[ChurnRequest]

class SalesForecastRequest(BaseModel):
    recent_sales: List[float] = Field(..., min_items=7, description="Last N days of sales values")
    forecast_days: int = Field(7, ge=1, le=30, description="Number of days to forecast")

# ── Lazy-loaded predictors ────────────────────────────────────────────────────

_churn_predictor = None
_sales_predictor = None

def get_churn_predictor():
    global _churn_predictor
    if _churn_predictor is None:
        from src.inference.predictor import ChurnPredictor
        _churn_predictor = ChurnPredictor()
    return _churn_predictor

def get_sales_predictor():
    global _sales_predictor
    if _sales_predictor is None:
        from src.inference.predictor import SalesPredictor
        _sales_predictor = SalesPredictor()
    return _sales_predictor

# ── Middleware ────────────────────────────────────────────────────────────────

@app.middleware("http")
async def track_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start) * 1000
    tracker.record_request(
        endpoint=str(request.url.path),
        status_code=response.status_code,
        latency_ms=latency_ms
    )
    return response

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {"message": "Advanced DS Project API", "version": "1.0.0", "status": "running"}

@app.get("/health", tags=["Info"])
def health():
    return {"status": "healthy", "models": ["churn", "sales"], "uptime_requests": tracker.total_requests}

@app.post("/churn/predict", tags=["Churn"])
def predict_churn(request: ChurnRequest):
    """Predict churn probability for a single customer."""
    try:
        predictor = get_churn_predictor()
        result = predictor.predict(request.dict())
        tracker.record_prediction("churn")
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Churn model not trained yet. Run train_all.py first.")
    except Exception as e:
        logger.error(f"Churn predict error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/churn/batch_predict", tags=["Churn"])
def batch_predict_churn(request: ChurnBatchRequest):
    """Predict churn for a batch of customers."""
    try:
        predictor = get_churn_predictor()
        results = predictor.predict_batch([c.dict() for c in request.customers])
        tracker.record_prediction("churn", count=len(results))
        return {"predictions": results, "count": len(results)}
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Churn model not trained yet.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/churn/metrics", tags=["Churn"])
def churn_metrics():
    """Get churn model evaluation metrics."""
    try:
        return get_churn_predictor().get_model_metrics()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Churn model not trained yet.")

@app.post("/sales/forecast", tags=["Sales"])
def forecast_sales(request: SalesForecastRequest):
    """Forecast sales for the next N days."""
    try:
        predictor = get_sales_predictor()
        result = predictor.forecast(request.recent_sales, request.forecast_days)
        tracker.record_prediction("sales")
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Sales model not trained yet. Run train_all.py first.")
    except Exception as e:
        logger.error(f"Sales forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sales/metrics", tags=["Sales"])
def sales_metrics():
    """Get sales model evaluation metrics."""
    try:
        return get_sales_predictor().get_model_metrics()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Sales model not trained yet.")

@app.get("/metrics", tags=["Monitoring"])
def api_metrics():
    """Get API usage and performance metrics."""
    return tracker.get_summary()
