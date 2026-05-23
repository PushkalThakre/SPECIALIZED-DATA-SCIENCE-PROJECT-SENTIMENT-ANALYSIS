"""
sales_model.py
LSTM neural network for time-series sales forecasting.
"""

import numpy as np
import os
import json
from loguru import logger

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. Using ARIMA fallback.")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

import joblib


class SalesModel:
    """
    Bidirectional LSTM for sales forecasting.
    Falls back to ARIMA if TensorFlow is unavailable.
    """

    def __init__(self, window_size: int = 7, use_deep_learning: bool = True):
        self.window_size = window_size
        self.use_deep_learning = use_deep_learning and TF_AVAILABLE
        self.model = None
        self.metrics = {}
        self.history = None

    def build(self):
        if self.use_deep_learning:
            model = Sequential([
                Bidirectional(LSTM(64, return_sequences=True),
                              input_shape=(self.window_size, 1)),
                Dropout(0.2),
                Bidirectional(LSTM(32)),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mse", metrics=["mae"])
            self.model = model
            logger.info(f"LSTM model built. Parameters: {model.count_params():,}")
        else:
            logger.info("Will use ARIMA fallback during training.")
        return self

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=16):
        if self.use_deep_learning:
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
            ]
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs, batch_size=batch_size,
                callbacks=callbacks, verbose=1
            )
        else:
            # ARIMA fallback: fit on training series
            train_series = y_train
            if STATSMODELS_AVAILABLE:
                self.model = ARIMA(train_series, order=(5, 1, 0)).fit()
            logger.info("ARIMA training complete.")
        return self

    def evaluate(self, X_test, y_test, preprocessor=None) -> dict:
        if self.use_deep_learning:
            y_pred = self.model.predict(X_test, verbose=0).flatten()
        else:
            y_pred = self.model.forecast(steps=len(y_test)) if self.model else np.zeros(len(y_test))

        if preprocessor:
            y_test_inv = preprocessor.inverse_transform(y_test)
            y_pred_inv = preprocessor.inverse_transform(y_pred)
        else:
            y_test_inv, y_pred_inv = y_test, y_pred

        mae = float(np.mean(np.abs(y_test_inv - y_pred_inv)))
        rmse = float(np.sqrt(np.mean((y_test_inv - y_pred_inv) ** 2)))
        mape = float(np.mean(np.abs((y_test_inv - y_pred_inv) / (y_test_inv + 1e-8))) * 100)

        self.metrics = {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 2)}
        logger.info(f"Evaluation — MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        return self.metrics

    def predict(self, X) -> dict:
        if self.use_deep_learning:
            pred = float(self.model.predict(X, verbose=0).flatten()[0])
        else:
            pred = float(self.model.forecast(steps=1)[0]) if self.model else 0.0
        return {"predicted_sales": round(pred, 4), "model_type": "LSTM" if self.use_deep_learning else "ARIMA"}

    def forecast_next_n(self, last_sequence: np.ndarray, n: int, preprocessor=None) -> list:
        forecasts = []
        seq = last_sequence.copy()
        for _ in range(n):
            if self.use_deep_learning:
                inp = seq[-self.window_size:].reshape(1, self.window_size, 1)
                pred = float(self.model.predict(inp, verbose=0)[0][0])
            else:
                pred = float(self.model.forecast(1)[0]) if self.model else 0.0
            forecasts.append(pred)
            seq = np.append(seq, pred)

        if preprocessor:
            forecasts = preprocessor.inverse_transform(np.array(forecasts)).tolist()
        return [round(v, 2) for v in forecasts]

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        if self.use_deep_learning and self.model:
            self.model.save(os.path.join(path, "sales_model.h5"))
        elif self.model:
            joblib.dump(self.model, os.path.join(path, "sales_model_arima.pkl"))
        with open(os.path.join(path, "sales_metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Sales model saved to {path}")

    @classmethod
    def load_saved(cls, path: str, window_size: int = 7, use_deep_learning: bool = True):
        obj = cls(window_size, use_deep_learning)
        if use_deep_learning and TF_AVAILABLE:
            obj.model = load_model(os.path.join(path, "sales_model.h5"))
        else:
            obj.model = joblib.load(os.path.join(path, "sales_model_arima.pkl"))
        metrics_path = os.path.join(path, "sales_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                obj.metrics = json.load(f)
        return obj
