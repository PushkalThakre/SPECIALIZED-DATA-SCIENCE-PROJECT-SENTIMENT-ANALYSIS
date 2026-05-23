"""
train_all.py
Master training script — trains both models sequentially.
Run: python -m src.training.train_all
"""

from loguru import logger
import json, os

def main():
    print("\n" + "=" * 60)
    print("  MONTH 5 — ADVANCED DATA SCIENCE PROJECT")
    print("  Training All Models")
    print("=" * 60)

    results = {}

    # Train Churn Model
    print("\n[1/2] Training Customer Churn Classification Model...")
    try:
        from src.training.train_churn import main as train_churn
        results["churn"] = train_churn()
    except Exception as e:
        logger.error(f"Churn training failed: {e}")
        results["churn"] = {"error": str(e)}

    # Train Sales Model
    print("\n[2/2] Training Sales Forecasting Model...")
    try:
        from src.training.train_sales import main as train_sales
        results["sales"] = train_sales()
    except Exception as e:
        logger.error(f"Sales training failed: {e}")
        results["sales"] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    if "churn" in results and "accuracy" in results.get("churn", {}):
        c = results["churn"]
        print(f"  Churn Model  — Accuracy: {c['accuracy']:.3f} | AUC: {c['roc_auc']:.3f}")
    if "sales" in results and "mae" in results.get("sales", {}):
        s = results["sales"]
        print(f"  Sales Model  — MAE: {s['mae']:.2f} | RMSE: {s['rmse']:.2f} | MAPE: {s['mape']:.2f}%")
    print("=" * 60)

    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/training_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Results saved to data/processed/training_summary.json")


if __name__ == "__main__":
    main()
