"""
sales_preprocessor.py
Preprocessing pipeline for Supermarket Sales time-series forecasting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from loguru import logger


class SalesPreprocessor:
    """Preprocessing for supermarket sales time-series data."""

    DATE_COL = "Date"
    TARGET_COL = "Total"

    def __init__(self, window_size: int = 7):
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.is_fitted = False

    def load(self, filepath: str) -> pd.DataFrame:
        logger.info(f"Loading sales data from {filepath}")
        df = pd.read_csv(filepath, parse_dates=[self.DATE_COL])
        logger.info(f"Loaded {len(df)} rows")
        return df

    def aggregate_daily(self, df: pd.DataFrame) -> pd.Series:
        """Aggregate transactions to daily total sales."""
        daily = df.groupby(self.DATE_COL)[self.TARGET_COL].sum().sort_index()
        # Fill missing dates
        full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        daily = daily.reindex(full_range, fill_value=0)
        logger.info(f"Daily series: {len(daily)} days from {daily.index.min().date()} to {daily.index.max().date()}")
        return daily

    def add_features(self, daily: pd.Series) -> pd.DataFrame:
        """Add calendar and lag features."""
        df = pd.DataFrame({"date": daily.index, "sales": daily.values})
        df["dayofweek"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
        df["lag_1"] = df["sales"].shift(1)
        df["lag_7"] = df["sales"].shift(7)
        df["rolling_7"] = df["sales"].shift(1).rolling(7).mean()
        df.dropna(inplace=True)
        return df

    def create_sequences(self, series: np.ndarray):
        """Create sliding window sequences for LSTM."""
        X, y = [], []
        for i in range(self.window_size, len(series)):
            X.append(series[i - self.window_size:i])
            y.append(series[i])
        return np.array(X), np.array(y)

    def fit_transform(self, filepath: str):
        df_raw = self.load(filepath)
        daily = self.aggregate_daily(df_raw)
        scaled = self.scaler.fit_transform(daily.values.reshape(-1, 1)).flatten()
        self.is_fitted = True
        X, y = self.create_sequences(scaled)
        X = X.reshape(X.shape[0], X.shape[1], 1)  # LSTM expects 3D
        logger.info(f"Sequences created: X={X.shape}, y={y.shape}")
        return X, y, daily

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(values.reshape(-1, 1)).flatten()

    def get_train_test_split(self, X, y, test_ratio=0.2):
        split = int(len(X) * (1 - test_ratio))
        return X[:split], X[split:], y[:split], y[split:]

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, os.path.join(path, "sales_preprocessor.pkl"))
        logger.info(f"Sales preprocessor saved to {path}")

    @classmethod
    def load_saved(cls, path: str):
        return joblib.load(os.path.join(path, "sales_preprocessor.pkl"))
