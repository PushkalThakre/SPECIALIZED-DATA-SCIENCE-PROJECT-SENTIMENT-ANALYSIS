"""
tracker.py
Lightweight in-memory metrics tracker for model monitoring.
"""

import time
import json
import os
from collections import defaultdict
from datetime import datetime
from loguru import logger


class MetricsTracker:
    """Tracks API requests, latency, prediction counts, and model drift signals."""

    def __init__(self, log_path: str = "monitoring/metrics.json"):
        self.log_path = log_path
        self.total_requests = 0
        self.error_count = 0
        self.latencies = []
        self.endpoint_counts = defaultdict(int)
        self.prediction_counts = defaultdict(int)
        self.start_time = time.time()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def record_request(self, endpoint: str, status_code: int, latency_ms: float):
        self.total_requests += 1
        self.endpoint_counts[endpoint] += 1
        self.latencies.append(latency_ms)
        if status_code >= 400:
            self.error_count += 1

    def record_prediction(self, model_name: str, count: int = 1):
        self.prediction_counts[model_name] += count

    def get_summary(self) -> dict:
        uptime = time.time() - self.start_time
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        error_rate = (self.error_count / self.total_requests * 100) if self.total_requests else 0

        summary = {
            "uptime_seconds": round(uptime, 1),
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "error_rate_pct": round(error_rate, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(sorted(self.latencies)[int(len(self.latencies)*0.95)] if len(self.latencies) > 20 else avg_latency, 2),
            "predictions_served": dict(self.prediction_counts),
            "endpoint_breakdown": dict(self.endpoint_counts),
            "timestamp": datetime.now().isoformat()
        }
        return summary

    def save_snapshot(self):
        summary = self.get_summary()
        try:
            with open(self.log_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Metrics snapshot saved to {self.log_path}")
        except IOError as e:
            logger.error(f"Could not save metrics: {e}")


class ModelDriftDetector:
    """Simple statistical drift detection comparing recent vs baseline predictions."""

    def __init__(self, baseline_mean: float, baseline_std: float, threshold: float = 2.0):
        self.baseline_mean = baseline_mean
        self.baseline_std = baseline_std
        self.threshold = threshold  # Z-score threshold
        self.recent_predictions = []

    def add_prediction(self, value: float):
        self.recent_predictions.append(value)
        if len(self.recent_predictions) > 100:
            self.recent_predictions.pop(0)

    def check_drift(self) -> dict:
        if len(self.recent_predictions) < 10:
            return {"drift_detected": False, "reason": "Insufficient data"}

        import statistics
        recent_mean = statistics.mean(self.recent_predictions)
        z_score = abs(recent_mean - self.baseline_mean) / (self.baseline_std + 1e-8)
        drift = z_score > self.threshold

        return {
            "drift_detected": drift,
            "z_score": round(z_score, 3),
            "recent_mean": round(recent_mean, 4),
            "baseline_mean": round(self.baseline_mean, 4),
            "alert": "⚠ DRIFT DETECTED — retrain recommended" if drift else "✔ No significant drift"
        }
