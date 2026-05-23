"""tests/test_models.py — Unit tests for model classes."""

import unittest
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.churn_model import ChurnModel
from src.models.sales_model import SalesModel


class TestChurnModel(unittest.TestCase):

    def setUp(self):
        self.model = ChurnModel(input_dim=7, use_deep_learning=False)
        self.model.build()
        np.random.seed(42)
        self.X_train = np.random.rand(80, 7)
        self.y_train = np.random.randint(0, 2, 80)
        self.X_test = np.random.rand(20, 7)
        self.y_test = np.random.randint(0, 2, 20)

    def test_build(self):
        self.assertIsNotNone(self.model.model)

    def test_train_and_predict(self):
        self.model.train(self.X_train, self.y_train, self.X_test, self.y_test)
        result = self.model.predict(self.X_test[:1])
        self.assertIn("churn_probability", result)
        self.assertIn("prediction", result)
        self.assertIn(result["prediction"], ["Churn", "No Churn"])
        self.assertGreaterEqual(result["churn_probability"], 0)
        self.assertLessEqual(result["churn_probability"], 1)

    def test_evaluate_returns_metrics(self):
        self.model.train(self.X_train, self.y_train, self.X_test, self.y_test)
        metrics = self.model.evaluate(self.X_test, self.y_test)
        for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            self.assertIn(key, metrics)

    def test_save_and_load(self):
        import tempfile, shutil
        self.model.train(self.X_train, self.y_train, self.X_test, self.y_test)
        tmpdir = tempfile.mkdtemp()
        try:
            self.model.save(tmpdir)
            loaded = ChurnModel.load_saved(tmpdir, input_dim=7, use_deep_learning=False)
            result = loaded.predict(self.X_test[:1])
            self.assertIn("churn_probability", result)
        finally:
            shutil.rmtree(tmpdir)


class TestSalesModel(unittest.TestCase):

    def setUp(self):
        self.window = 5
        self.model = SalesModel(window_size=self.window, use_deep_learning=False)
        np.random.seed(42)
        n = 60
        self.X_train = np.random.rand(n, self.window, 1)
        self.y_train = np.random.rand(n)
        self.X_test = np.random.rand(10, self.window, 1)
        self.y_test = np.random.rand(10)

    def test_predict_returns_dict(self):
        self.model.build()
        self.model.train(self.X_train, self.y_train, self.X_test, self.y_test)
        result = self.model.predict(self.X_test[:1])
        self.assertIn("predicted_sales", result)

    def test_evaluate_keys(self):
        self.model.build()
        self.model.train(self.X_train, self.y_train, self.X_test, self.y_test)
        metrics = self.model.evaluate(self.X_test, self.y_test)
        for key in ["mae", "rmse", "mape"]:
            self.assertIn(key, metrics)


if __name__ == "__main__":
    unittest.main()
