"""Training script for RGB+Thermal multimodal experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.training.train_yolo import run

if __name__ == "__main__":
    run(modality="rgb", model_name="yolov8s", epochs=100, batch_size=8, img_size=640)
