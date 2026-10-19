"""
Evaluation script for Thermal-only detection model.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.evaluation.evaluate_yolo import evaluate
from src.config import MODEL_NAME


def run_evaluation(model_name: str = MODEL_NAME, img_size: int = 320, batch_size: int = 16):
    return evaluate(modality="thermal", model_name=model_name, img_size=img_size, batch_size=batch_size)


if __name__ == "__main__":
    run_evaluation()
