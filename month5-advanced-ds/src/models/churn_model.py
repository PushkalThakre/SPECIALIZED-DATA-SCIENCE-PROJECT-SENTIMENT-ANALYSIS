"""
churn_model.py
Deep learning model for customer churn classification.
"""

import numpy as np
import os
import json
from loguru import logger

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not installed. Using scikit-learn fallback.")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib


class ChurnModel:
    """
    Neural network for binary churn classification.
    Falls back to GradientBoosting if TensorFlow is unavailable.
    """

    def __init__(self, input_dim: int, use_deep_learning: bool = True):
        self.input_dim = input_dim
        self.use_deep_learning = use_deep_learning and TF_AVAILABLE
        self.model = None
        self.history = None
        self.metrics = {}

    def build(self):
        if self.use_deep_learning:
            model = Sequential([
                Dense(128, activation="relu", input_shape=(self.input_dim,)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(64, activation="relu"),
                BatchNormalization(),
                Dropout(0.2),
                Dense(32, activation="relu"),
                Dropout(0.1),
                Dense(1, activation="sigmoid")
            ])
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss="binary_crossentropy",
                metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
            )
            self.model = model
            logger.info(f"Deep learning model built. Parameters: {model.count_params():,}")
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05,
                max_depth=4, random_state=42
            )
            logger.info("Gradient Boosting fallback model created.")
        return self

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        if self.use_deep_learning:
            callbacks = [
                EarlyStopping(monitor="val_auc", patience=10, restore_best_weights=True, mode="max"),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
            ]
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            logger.info("Deep learning training complete.")
        else:
            self.model.fit(X_train, y_train)
            logger.info("Gradient Boosting training complete.")
        return self

    def evaluate(self, X_test, y_test) -> dict:
        if self.use_deep_learning:
            y_prob = self.model.predict(X_test).flatten()
            y_pred = (y_prob > 0.5).astype(int)
        else:
            y_prob = self.model.predict_proba(X_test)[:, 1]
            y_pred = self.model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        self.metrics = {
            "accuracy": report["accuracy"],
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1_score": report["weighted avg"]["f1-score"],
            "roc_auc": auc,
            "confusion_matrix": cm.tolist()
        }
        logger.info(f"Evaluation — Accuracy: {self.metrics['accuracy']:.4f}, AUC: {auc:.4f}")
        return self.metrics

    def predict(self, X) -> dict:
        if self.use_deep_learning:
            prob = float(self.model.predict(X, verbose=0).flatten()[0])
        else:
            prob = float(self.model.predict_proba(X)[:, 1][0])
        return {
            "churn_probability": round(prob, 4),
            "prediction": "Churn" if prob > 0.5 else "No Churn",
            "confidence": round(max(prob, 1 - prob) * 100, 1)
        }

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        if self.use_deep_learning:
            self.model.save(os.path.join(path, "churn_model.h5"))
        else:
            joblib.dump(self.model, os.path.join(path, "churn_model.pkl"))
        with open(os.path.join(path, "churn_metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Churn model saved to {path}")

    @classmethod
    def load_saved(cls, path: str, input_dim: int, use_deep_learning: bool = True):
        obj = cls(input_dim, use_deep_learning)
        if use_deep_learning and TF_AVAILABLE:
            obj.model = load_model(os.path.join(path, "churn_model.h5"))
        else:
            obj.model = joblib.load(os.path.join(path, "churn_model.pkl"))
        metrics_path = os.path.join(path, "churn_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                obj.metrics = json.load(f)
        logger.info("Churn model loaded.")
        return obj
