"""
predictor.py
Unified prediction interface for both models.
"""

import numpy as np
import os
import json
from loguru import logger
from typing import Dict, Any, List


class ChurnPredictor:
    """Real-time churn prediction from raw customer data."""

    def __init__(self, model_path: str = "data/processed/churn_model"):
        self.model_path = model_path
        self._model = None
        self._preprocessor = None

    def _load(self):
        if self._model is None:
            from src.models.churn_model import ChurnModel
            from src.data_processing.churn_preprocessor import ChurnPreprocessor
            self._preprocessor = ChurnPreprocessor.load_saved(self.model_path)
            n_features = len(self._preprocessor.feature_names)
            self._model = ChurnModel.load_saved(self.model_path, input_dim=n_features)
            logger.info("ChurnPredictor loaded.")

    def predict(self, customer_data: dict) -> Dict[str, Any]:
        self._load()
        import pandas as pd
        df = pd.DataFrame([customer_data])
        X = self._preprocessor.transform(df)
        result = self._model.predict(X)
        result["input"] = customer_data
        return result

    def predict_batch(self, customers: List[dict]) -> List[dict]:
        return [self.predict(c) for c in customers]

    def get_model_metrics(self) -> dict:
        self._load()
        return self._model.metrics


class SalesPredictor:
    """Real-time sales forecasting."""

    def __init__(self, model_path: str = "data/processed/sales_model", window_size: int = 7):
        self.model_path = model_path
        self.window_size = window_size
        self._model = None
        self._preprocessor = None

    def _load(self):
        if self._model is None:
            from src.models.sales_model import SalesModel
            from src.data_processing.sales_preprocessor import SalesPreprocessor
            self._preprocessor = SalesPreprocessor.load_saved(self.model_path)
            self._model = SalesModel.load_saved(self.model_path, window_size=self.window_size)
            logger.info("SalesPredictor loaded.")

    def forecast(self, recent_sales: List[float], n_days: int = 7) -> Dict[str, Any]:
        self._load()
        series = np.array(recent_sales)
        scaled = self._preprocessor.scaler.transform(series.reshape(-1, 1)).flatten()
        forecasts = self._model.forecast_next_n(scaled, n=n_days, preprocessor=self._preprocessor)
        return {
            "forecast_days": n_days,
            "predicted_sales": forecasts,
            "total_predicted": round(sum(forecasts), 2)
        }

    def get_model_metrics(self) -> dict:
        self._load()
        return self._model.metrics
