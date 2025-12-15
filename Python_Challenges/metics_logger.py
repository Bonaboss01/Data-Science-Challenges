"""
A lightweight utility for tracking model metrics over time.
Designed for continuous experimentation and reproducible ML workflows.
"""

from datetime import datetime
import json
import os
from typing import Dict


LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "metrics_history.jsonl")


def log_metrics(
    model_name: str,
    metrics: Dict[str, float],
    notes: str = ""
) -> None:
    """
    Append model metrics to a JSON Lines file.

    Args:
        model_name (str): Name of the model or experiment
        metrics (dict): Dictionary of metric names and values
        notes (str): Optional notes about the run
    """

    os.makedirs(LOG_DIR, exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model_name,
        "metrics": metrics,
        "notes": notes
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    # Example usage (safe to modify & extend over time)
    log_metrics(
        model_name="baseline_linear_regression",
        metrics={
            "rmse": 12.84,
            "mae": 9.31,
            "r2": 0.78
        },
        notes="Initial baseline using time-based split"
    )

    print("Metrics logged successfully.")
