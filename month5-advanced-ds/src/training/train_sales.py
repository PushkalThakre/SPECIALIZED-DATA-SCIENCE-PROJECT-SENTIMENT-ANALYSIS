"""
train_sales.py
Full training pipeline for the LSTM sales forecasting model.
Run: python -m src.training.train_sales
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger

from src.data_processing.sales_preprocessor import SalesPreprocessor
from src.models.sales_model import SalesModel

DATA_PATH = "data/raw/supermarket_sales.csv"
MODEL_SAVE_PATH = "data/processed/sales_model"
REPORTS_PATH = "data/processed"


def plot_forecast_vs_actual(y_true, y_pred, preprocessor, save_path: str):
    y_true_inv = preprocessor.inverse_transform(y_true)
    y_pred_inv = preprocessor.inverse_transform(y_pred)
    plt.figure(figsize=(14, 5))
    plt.plot(y_true_inv, label="Actual Sales", color="steelblue")
    plt.plot(y_pred_inv, label="Predicted Sales", color="orange", linestyle="--")
    plt.title("Sales Forecasting — Actual vs Predicted")
    plt.xlabel("Day"); plt.ylabel("Total Sales (₹)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_path, "sales_forecast_plot.png"), dpi=100)
    plt.close()
    logger.info("Forecast plot saved.")


def print_results(metrics: dict):
    print("\n" + "=" * 55)
    print("  SALES FORECASTING MODEL — RESULTS")
    print("=" * 55)
    print(f"  MAE   (Mean Absolute Error)   : {metrics['mae']:.2f}")
    print(f"  RMSE  (Root Mean Square Error): {metrics['rmse']:.2f}")
    print(f"  MAPE  (Mean Abs % Error)      : {metrics['mape']:.2f}%")
    print("=" * 55)


def main():
    logger.info("=== SALES TRAINING PIPELINE START ===")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(REPORTS_PATH, exist_ok=True)

    # 1. Preprocess
    preprocessor = SalesPreprocessor(window_size=7)
    X, y, daily_series = preprocessor.fit_transform(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocessor.get_train_test_split(X, y)
    split = int(len(X_train) * 0.9)
    X_tr, X_val = X_train[:split], X_train[split:]
    y_tr, y_val = y_train[:split], y_train[split:]
    logger.info(f"Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 2. Build & Train
    model = SalesModel(window_size=7)
    model.build()
    model.train(X_tr, y_tr, X_val, y_val, epochs=50, batch_size=16)

    # 3. Evaluate
    metrics = model.evaluate(X_test, y_test, preprocessor)
    print_results(metrics)

    # 4. Plot predictions
    if model.use_deep_learning:
        y_pred = model.model.predict(X_test, verbose=0).flatten()
        plot_forecast_vs_actual(y_test, y_pred, preprocessor, REPORTS_PATH)

    # 5. Sample future forecast
    last_seq = X_test[-1].flatten()
    forecast = model.forecast_next_n(last_seq, n=7, preprocessor=preprocessor)
    print(f"\n  7-Day Forecast: {forecast}")

    # 6. Save
    model.save(MODEL_SAVE_PATH)
    preprocessor.save(MODEL_SAVE_PATH)

    logger.info("=== SALES TRAINING PIPELINE COMPLETE ===")
    return metrics


if __name__ == "__main__":
    main()
