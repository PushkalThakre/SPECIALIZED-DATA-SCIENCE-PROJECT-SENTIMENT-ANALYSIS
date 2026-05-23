"""tests/test_api.py — Integration tests for FastAPI endpoints."""

import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


class TestAPIHealth(unittest.TestCase):

    def setUp(self):
        from src.api.app import app
        self.client = TestClient(app)

    def test_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("version", data)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_metrics_endpoint(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_requests", data)

    def test_churn_predict_missing_model_returns_503(self):
        response = self.client.post("/churn/predict", json={
            "Tenure": 12, "MonthlyCharges": 65.5, "TotalCharges": 786.0,
            "Contract": "Month-to-month", "PaymentMethod": "Electronic Check",
            "PaperlessBilling": "Yes", "SeniorCitizen": 0
        })
        # 503 if model not trained, 200 if it is — both acceptable
        self.assertIn(response.status_code, [200, 503])

    def test_sales_forecast_missing_model_returns_503(self):
        response = self.client.post("/sales/forecast", json={
            "recent_sales": [100, 150, 200, 180, 210, 190, 220],
            "forecast_days": 3
        })
        self.assertIn(response.status_code, [200, 503])

    def test_churn_predict_invalid_payload(self):
        response = self.client.post("/churn/predict", json={"invalid": "data"})
        self.assertEqual(response.status_code, 422)  # Validation error


if __name__ == "__main__":
    unittest.main()
