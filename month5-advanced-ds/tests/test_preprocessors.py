"""tests/test_preprocessors.py — Unit tests for data preprocessing."""

import unittest
import sys, os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_processing.churn_preprocessor import ChurnPreprocessor
from src.data_processing.sales_preprocessor import SalesPreprocessor


class TestChurnPreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = ChurnPreprocessor()
        self.df = pd.DataFrame({
            "CustomerID": ["C001", "C002", "C003"],
            "Tenure": [12, 24, 6],
            "MonthlyCharges": [65.0, 80.0, 45.0],
            "TotalCharges": [780.0, 1920.0, 270.0],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaymentMethod": ["Electronic Check", "Credit Card", "Bank Transfer"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "SeniorCitizen": [0, 1, 0],
            "Churn": [1, 0, 0]
        })

    def test_fit_transform_shape(self):
        X, y = self.preprocessor.fit_transform(self.df)
        self.assertEqual(X.shape[0], 3)
        self.assertEqual(len(y), 3)

    def test_no_nan_after_transform(self):
        X, y = self.preprocessor.fit_transform(self.df)
        self.assertFalse(np.isnan(X).any())

    def test_target_binary(self):
        _, y = self.preprocessor.fit_transform(self.df)
        self.assertTrue(set(y).issubset({0, 1}))

    def test_transform_after_fit(self):
        self.preprocessor.fit_transform(self.df)
        df_no_target = self.df.drop(columns=["Churn"])
        X = self.preprocessor.transform(df_no_target)
        self.assertEqual(X.shape[0], 3)

    def test_missing_values_handled(self):
        df = self.df.copy()
        df.loc[0, "MonthlyCharges"] = None
        X, y = self.preprocessor.fit_transform(df)
        self.assertFalse(np.isnan(X).any())


class TestSalesPreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = SalesPreprocessor(window_size=3)

    def test_aggregate_daily(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "Date": dates.repeat(2),
            "Total": np.random.uniform(100, 500, 20)
        })
        daily = self.preprocessor.aggregate_daily(df)
        self.assertEqual(len(daily), 10)

    def test_create_sequences(self):
        series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        X, y = self.preprocessor.create_sequences(series)
        self.assertEqual(X.shape[0], len(series) - self.preprocessor.window_size)
        self.assertEqual(X.shape[1], self.preprocessor.window_size)


if __name__ == "__main__":
    unittest.main()
