"""Unified evaluation for all YOLO experiments.

Evaluates a trained YOLO model on test set and calculates:
- Precision, Recall, F1-score
- mAP@50, mAP@50:95
- Inference time
"""

import csv
import json
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    DATA_VISIBLE_TEST, DATA_INFRARED_TEST,
    OUTPUT_RGB, OUTPUT_THERMAL, OUTPUT_MULTIMODAL,
    SEED,
)
from ultralytics import YOLO


def get_test_dir(modality: str) -> Path:
    if modality == "thermal":
        return DATA_INFRARED_TEST
    return DATA_VISIBLE_TEST


def get_output_base(modality: str, model_name: str) -> Path:
    if modality == "rgb":
        return OUTPUT_RGB / model_name
    elif modality == "thermal":
        return OUTPUT_THERMAL / model_name
    else:
        return OUTPUT_MULTIMODAL / model_name


def evaluate(modality: str, model_name: str, img_size: int = 640, batch_size: int = 16) -> dict:
    """Evaluate a YOLO model and return metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating {modality.upper()} + {model_name}")
    print(f"{'='*60}")

    output_base = get_output_base(modality, model_name)
    test_dir = get_test_dir(modality)

    model_path = output_base / "train" / "weights" / "best.pt"
    if not model_path.exists():
        model_path = output_base / "train" / "weights" / "last.pt"
    if not model_path.exists():
        print(f"ERROR: No model weights found in {output_base / 'train' / 'weights'}")
        return {}

    model = YOLO(str(model_path))

    results = model.val(
        data=str(output_base / "yolo_dataset" / "data.yaml"),
        split="test",
        imgsz=img_size,
        batch=batch_size,
        device="cpu",
        workers=0,
        save_json=True,
        save=False,
        conf=0.25,
        iou=0.5,
    )

    metrics = results.results_dict if hasattr(results, "results_dict") else {}

    precision = metrics.get("metrics/precision(B)", None)
    recall = metrics.get("metrics/recall(B)", None)
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )

    metrics_dict = {
        "modality": modality,
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mAP50": metrics.get("metrics/mAP50(B)", None),
        "mAP50_95": metrics.get("metrics/mAP50-95(B)", None),
        "fitness": metrics.get("fitness", None),
        "inference_time": None,
    }

    print(f"\nTest Set Metrics ({modality.upper()} + {model_name}):")
    print(f"  Precision:  {metrics_dict['precision']}")
    print(f"  Recall:     {metrics_dict['recall']}")
    print(f"  F1-score:   {metrics_dict['f1_score']}")
    print(f"  mAP@50:     {metrics_dict['mAP50']}")
    print(f"  mAP@50:95:  {metrics_dict['mAP50_95']}")

    # Measure inference time on a small sample
    image_files = sorted([f for f in test_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    sample = image_files[:100] if len(image_files) > 100 else image_files

    times = []
    for img_path in sample:
        t0 = time.time()
        model.predict(str(img_path), imgsz=img_size, verbose=False, conf=0.25)
        t1 = time.time()
        times.append(t1 - t0)

    if times:
        avg_ms = sum(times) / len(times) * 1000
        metrics_dict["inference_time"] = {
            "avg_per_image_ms": round(avg_ms, 2),
            "total_ms": round(sum(times) * 1000, 2),
            "samples": len(times),
        }
        print(f"  Inference time (avg over {len(times)} images): {avg_ms:.2f} ms")

    # Save results
    results_path = output_base / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics_dict, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Save as CSV for easy comparison
    csv_path = output_base / "test_results.csv"
    flat = {k: v for k, v in metrics_dict.items() if not isinstance(v, dict)}
    if metrics_dict.get("inference_time"):
        flat.update(metrics_dict["inference_time"])
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat.keys())
        writer.writeheader()
        writer.writerow(flat)
    print(f"CSV results saved to: {csv_path}")

    return metrics_dict
