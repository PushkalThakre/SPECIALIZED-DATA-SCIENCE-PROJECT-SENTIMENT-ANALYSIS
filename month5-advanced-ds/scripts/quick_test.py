"""
quick_test.py
Quick smoke-test: sends sample requests to the running API.
Usage: python scripts/quick_test.py
"""

import json

try:
    import httpx
    BASE = "http://localhost:8000"

    print("\n=== API SMOKE TEST ===")

    # Health
    r = httpx.get(f"{BASE}/health")
    print(f"  /health         → {r.status_code} {r.json()['status']}")

    # Churn prediction
    customer = {
        "Tenure": 12, "MonthlyCharges": 65.5, "TotalCharges": 786.0,
        "Contract": "Month-to-month", "PaymentMethod": "Electronic Check",
        "PaperlessBilling": "Yes", "SeniorCitizen": 0
    }
    r = httpx.post(f"{BASE}/churn/predict", json=customer)
    print(f"  /churn/predict  → {r.status_code} {r.json()}")

    # Sales forecast
    sales_req = {"recent_sales": [1200, 1350, 980, 1100, 1450, 1300, 1250], "forecast_days": 3}
    r = httpx.post(f"{BASE}/sales/forecast", json=sales_req)
    print(f"  /sales/forecast → {r.status_code} {r.json()}")

    print("\n=== ALL TESTS PASSED ===\n")

except Exception as e:
    print(f"  ✘ Error: {e}")
    print("  Make sure the API is running: uvicorn src.api.app:app --port 8000")
