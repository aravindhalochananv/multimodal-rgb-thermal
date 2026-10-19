"""Training script for RGB-only experiment."""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
from src.training.train_yolo import run

if __name__ == "__main__":
    run(modality="rgb", model_name="yolov8n", epochs=100, batch_size=8, img_size=640)
