"""
Evaluation module for trained models.
Calculates precision, recall, F1, mAP@50, mAP@50:95, inference time.
"""

import csv
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

from ultralytics import YOLO


def find_best_weights(output_dir: Path) -> Path:
    """Find best.pt or last.pt in training output directory."""
    weights_dir = output_dir / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    if best.exists():
        return best
    elif last.exists():
        return last
    else:
        raise FileNotFoundError(f"No weights found in {weights_dir}")


def evaluate_model(model_path: Path, data_yaml: Path, test_images_dir: Path,
                   imgsz: int = 640, batch: int = 16, device: str = "0",
                   conf: float = 0.25, iou: float = 0.5,
                   num_inference_samples: int = 100) -> Dict:
    """Evaluate a YOLO model on test set."""
    model = YOLO(str(model_path))

    # Validation metrics
    results = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=4,
        conf=conf,
        iou=iou,
        save_json=True,
        save=False,
    )

    d = results.results_dict
    precision = d.get("metrics/precision(B)")
    recall = d.get("metrics/recall(B)")
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) > 0 else None)

    metrics = {
        "model_path": str(model_path),
        "timestamp": datetime.now().isoformat(),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mAP50": d.get("metrics/mAP50(B)"),
        "mAP50_95": d.get("metrics/mAP50-95(B)"),
        "fitness": d.get("fitness"),
    }

    # Inference time measurement
    image_files = sorted(f for f in test_images_dir.iterdir()
                         if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    sample = image_files[:num_inference_samples]

    times = []
    for img_path in sample:
        t0 = time.time()
        model.predict(str(img_path), imgsz=imgsz, verbose=False, conf=conf)
        times.append(time.time() - t0)

    if times:
        avg_ms = sum(times) / len(times) * 1000
        metrics["inference_time"] = {
            "avg_per_image_ms": round(avg_ms, 2),
            "total_ms": round(sum(times) * 1000, 2),
            "samples": len(times),
            "device": device,
        }
        print(f"  Inference time (avg over {len(times)} images): {avg_ms:.2f} ms")

    return metrics


def evaluate_fusion_early(model_path: Path, data_yaml: Path,
                          test_rgb_dir: Path, test_thermal_dir: Path,
                          imgsz: int = 640, batch: int = 16, device: str = "0",
                          conf: float = 0.25, iou: float = 0.5,
                          num_inference_samples: int = 100) -> Dict:
    """Evaluate early fusion model (requires custom dataset)."""
    from models import EarlyFusionYOLO
    import torch
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    import numpy as np

    model = EarlyFusionYOLO("yolov8n", num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    model.eval()

    # Custom dataset
    class FusionTestDataset(Dataset):
        def __init__(self, rgb_dir, thermal_dir, label_dir, imgsz=640):
            self.rgb_dir = Path(rgb_dir)
            self.thermal_dir = Path(thermal_dir)
            self.label_dir = Path(label_dir)
            self.imgsz = imgsz
            self.files = sorted(f for f in self.rgb_dir.iterdir()
                                if f.suffix.lower() in {".jpg", ".jpeg", ".png"})

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            f = self.files[idx]
            stem = f.stem

            rgb = Image.open(f).convert("RGB").resize((self.imgsz, self.imgsz))
            rgb = np.array(rgb).transpose(2, 0, 1) / 255.0

            thermal_f = self.thermal_dir / f.name
            thermal = Image.open(thermal_f).convert("RGB").resize((self.imgsz, self.imgsz))
            thermal = np.array(thermal).transpose(2, 0, 1) / 255.0

            x = np.concatenate([rgb, thermal], axis=0).astype(np.float32)

            label_f = self.label_dir / f"{stem}.txt"
            if label_f.exists():
                labels = np.loadtxt(label_f).reshape(-1, 5)
            else:
                labels = np.zeros((0, 5), dtype=np.float32)

            return torch.from_numpy(x), torch.from_numpy(labels), stem

    dataset = FusionTestDataset(test_rgb_dir, test_thermal_dir,
                                Path(data_yaml).parent / "labels" / "test", imgsz)
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=4)

    # Run validation
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, targets, _ in loader:
            x = x.to(device)
            preds = model(x[:, :3], x[:, 3:])  # Split into rgb, thermal
            all_preds.append(preds.cpu())
            all_targets.append(targets)

    # Compute metrics using ultralytics val
    # For simplicity, use standard YOLO val on the fusion dataset
    from ultralytics import YOLO
    std_model = YOLO(str(model_path))
    results = std_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=4,
        conf=conf,
        iou=iou,
    )

    d = results.results_dict
    precision = d.get("metrics/precision(B)")
    recall = d.get("metrics/recall(B)")
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) > 0 else None)

    metrics = {
        "model_path": str(model_path),
        "timestamp": datetime.now().isoformat(),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mAP50": d.get("metrics/mAP50(B)"),
        "mAP50_95": d.get("metrics/mAP50-95(B)"),
    }

    return metrics


def save_results(metrics: Dict, output_dir: Path) -> Tuple[Path, Path]:
    """Save evaluation results as JSON and CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "test_results.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    csv_path = output_dir / "test_results.csv"
    flat = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    if metrics.get("inference_time"):
        flat.update(metrics["inference_time"])
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat.keys())
        writer.writeheader()
        writer.writerow(flat)

    return json_path, csv_path


def compare_models(metrics_list: List[Dict], output_dir: Path) -> Dict:
    """Compare multiple model metrics and generate comparison table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build comparison table
    rows = []
    for m in metrics_list:
        row = {
            "model": m.get("model_path", "unknown"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "f1_score": m.get("f1_score"),
            "mAP50": m.get("mAP50"),
            "mAP50_95": m.get("mAP50_95"),
        }
        if m.get("inference_time"):
            row["inference_ms"] = m["inference_time"].get("avg_per_image_ms")
        rows.append(row)

    # Save CSV
    csv_path = output_dir / "comparison.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    # Save JSON
    json_path = output_dir / "comparison.json"
    with open(json_path, "w") as f:
        json.dump({"comparison": rows}, f, indent=2, default=str)

    # Print table
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    headers = ["Model", "Precision", "Recall", "F1", "mAP@50", "mAP@50:95", "Inference (ms)"]
    print(f"{headers[0]:<30} {headers[1]:>10} {headers[2]:>10} {headers[3]:>8} {headers[4]:>8} {headers[5]:>10} {headers[6]:>12}")
    print("-"*80)
    for r in rows:
        model_name = Path(r["model"]).parent.parent.name
        print(f"{model_name:<30} {r['precision']:>10.4f} {r['recall']:>10.4f} "
              f"{r['f1_score']:>8.4f} {r['mAP50']:>8.4f} {r['mAP50_95']:>10.4f} "
              f"{r.get('inference_ms', 0):>12.2f}")

    return {"csv": csv_path, "json": json_path, "rows": rows}


def evaluate_all(modality: str = "all") -> List[Dict]:
    """Evaluate all trained models."""
    from configs.config import (
        OUTPUT_RGB, OUTPUT_THERMAL, OUTPUT_FUSION,
        YOLO_RGB_DIR, YOLO_THERMAL_DIR, YOLO_FUSION_DIR,
        VISIBLE_TEST, INFRARED_TEST, IMG_SIZE, BATCH_SIZE, DEVICE
    )

    results = []

    if modality in ("rgb", "all"):
        print("\n>>> Evaluating RGB model...")
        try:
            best = find_best_weights(OUTPUT_RGB / "yolov8n" / "train")
            metrics = evaluate_model(best, YOLO_RGB_DIR / "data.yaml", VISIBLE_TEST,
                                     imgsz=IMG_SIZE, batch=BATCH_SIZE, device=DEVICE)
            metrics["modality"] = "rgb"
            save_results(metrics, OUTPUT_RGB / "yolov8n")
            results.append(metrics)
            print(f"  mAP@50: {metrics['mAP50']:.4f}, mAP@50:95: {metrics['mAP50_95']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")

    if modality in ("thermal", "all"):
        print("\n>>> Evaluating Thermal model...")
        try:
            best = find_best_weights(OUTPUT_THERMAL / "yolov8n" / "train")
            metrics = evaluate_model(best, YOLO_THERMAL_DIR / "data.yaml", INFRARED_TEST,
                                     imgsz=IMG_SIZE, batch=BATCH_SIZE, device=DEVICE)
            metrics["modality"] = "thermal"
            save_results(metrics, OUTPUT_THERMAL / "yolov8n")
            results.append(metrics)
            print(f"  mAP@50: {metrics['mAP50']:.4f}, mAP@50:95: {metrics['mAP50_95']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")

    if modality in ("fusion", "all"):
        print("\n>>> Evaluating Fusion model...")
        try:
            best = find_best_weights(OUTPUT_FUSION / "yolov8n" / "train")
            metrics = evaluate_fusion_early(best, YOLO_FUSION_DIR / "data.yaml",
                                            VISIBLE_TEST, INFRARED_TEST,
                                            imgsz=IMG_SIZE, batch=BATCH_SIZE, device=DEVICE)
            metrics["modality"] = "fusion"
            save_results(metrics, OUTPUT_FUSION / "yolov8n")
            results.append(metrics)
            print(f"  mAP@50: {metrics['mAP50']:.4f}, mAP@50:95: {metrics['mAP50_95']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")

    if results:
        compare_models(results, Path(OUTPUT_RGB).parent / "comparison")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--modality", type=str, default="all",
                        choices=["rgb", "thermal", "fusion", "all"])
    args = parser.parse_args()

    evaluate_all(args.modality)