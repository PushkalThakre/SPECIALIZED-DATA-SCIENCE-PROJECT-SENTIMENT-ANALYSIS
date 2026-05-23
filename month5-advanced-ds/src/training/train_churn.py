"""
train_churn.py
Full training pipeline for the customer churn model.
Run: python -m src.training.train_churn
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger

from src.data_processing.churn_preprocessor import ChurnPreprocessor
from src.models.churn_model import ChurnModel

DATA_PATH = "data/raw/customer_churn.csv"
MODEL_SAVE_PATH = "data/processed/churn_model"
REPORTS_PATH = "data/processed"


def plot_training_history(history, save_path: str):
    if history is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss Curve"); axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="Train Acc")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc")
    axes[1].set_title("Accuracy Curve"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "churn_training_curves.png"), dpi=100)
    plt.close()
    logger.info("Training curves saved.")


def print_results(metrics: dict):
    print("\n" + "=" * 55)
    print("  CHURN MODEL — EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Accuracy   : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  F1-Score   : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion Matrix:")
    print(f"              Predicted No  Predicted Yes")
    print(f"  Actual No      {cm[0][0]:5d}         {cm[0][1]:5d}")
    print(f"  Actual Yes     {cm[1][0]:5d}         {cm[1][1]:5d}")
    print("=" * 55)


def main():
    logger.info("=== CHURN TRAINING PIPELINE START ===")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(REPORTS_PATH, exist_ok=True)

    # 1. Preprocess
    preprocessor = ChurnPreprocessor()
    X, y = preprocessor.fit_transform(preprocessor.load(DATA_PATH))
    X_train, X_test, y_train, y_test = preprocessor.get_train_test_split(X, y)
    X_train, X_val, y_train, y_val = preprocessor.get_train_test_split(X_train, y_train, test_size=0.1)
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 2. Build & Train
    model = ChurnModel(input_dim=X_train.shape[1])
    model.build()
    model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

    # 3. Evaluate
    metrics = model.evaluate(X_test, y_test)
    print_results(metrics)

    # 4. Save
    model.save(MODEL_SAVE_PATH)
    preprocessor.save(MODEL_SAVE_PATH)
    plot_training_history(model.history, REPORTS_PATH)

    logger.info("=== CHURN TRAINING PIPELINE COMPLETE ===")
    return metrics


if __name__ == "__main__":
    main()
