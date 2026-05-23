"""
churn_preprocessor.py
Preprocessing pipeline for Customer Churn dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from loguru import logger


class ChurnPreprocessor:
    """Full preprocessing pipeline for customer churn data."""

    CATEGORICAL_COLS = ["Contract", "PaymentMethod", "PaperlessBilling"]
    NUMERIC_COLS = ["Tenure", "MonthlyCharges", "TotalCharges"]
    TARGET_COL = "Churn"
    DROP_COLS = ["CustomerID"]

    def __init__(self):
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False

    def load(self, filepath: str) -> pd.DataFrame:
        logger.info(f"Loading churn data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Drop ID columns
        df.drop(columns=[c for c in self.DROP_COLS if c in df.columns], inplace=True)
        # Coerce numeric
        for col in self.NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Fill missing
        df[self.NUMERIC_COLS] = df[self.NUMERIC_COLS].fillna(df[self.NUMERIC_COLS].median())
        df[self.CATEGORICAL_COLS] = df[self.CATEGORICAL_COLS].fillna("Unknown")
        logger.info(f"After cleaning: {df.shape}, missing={df.isnull().sum().sum()}")
        return df

    def fit_transform(self, df: pd.DataFrame):
        df = self.clean(df)
        X = df.drop(columns=[self.TARGET_COL])
        y = df[self.TARGET_COL].values

        # Encode categoricals
        for col in self.CATEGORICAL_COLS:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le

        # Scale numerics
        X[self.NUMERIC_COLS] = self.scaler.fit_transform(X[self.NUMERIC_COLS])
        self.feature_names = list(X.columns)
        self.is_fitted = True
        logger.info(f"Fit complete. Features: {self.feature_names}")
        return X.values, y

    def transform(self, df: pd.DataFrame):
        if not self.is_fitted:
            raise RuntimeError("Call fit_transform first.")
        df = self.clean(df)
        if self.TARGET_COL in df.columns:
            df = df.drop(columns=[self.TARGET_COL])
        for col, le in self.encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))
        df[self.NUMERIC_COLS] = self.scaler.transform(df[self.NUMERIC_COLS])
        return df[self.feature_names].values

    def get_train_test_split(self, X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, os.path.join(path, "churn_preprocessor.pkl"))
        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load_saved(cls, path: str):
        obj = joblib.load(os.path.join(path, "churn_preprocessor.pkl"))
        logger.info("Preprocessor loaded.")
        return obj
