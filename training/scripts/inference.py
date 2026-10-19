"""
Inference script for trained models.
Supports RGB, thermal, and fusion models.
"""

import argparse
from pathlib import Path
import sys
import cv2
import numpy as np

from ultralytics import YOLO


def load_model(weights_path: Path):
    """Load YOLO model from weights."""
    return YOLO(str(weights_path))


def run_inference_rgb(model, image_path: Path, imgsz: int = 640, conf: float = 0.25):
    """Run inference on RGB image."""
    results = model.predict(str(image_path), imgsz=imgsz, conf=conf, verbose=False)
    return results[0]


def run_inference_thermal(model, image_path: Path, imgsz: int = 640, conf: float = 0.25):
    """Run inference on thermal image (same as RGB for single modality)."""
    return run_inference_rgb(model, image_path, imgsz, conf)


def run_inference_fusion(rgb_model, thermal_model, rgb_path: Path, thermal_path: Path,
                         imgsz: int = 640, conf: float = 0.25, iou: float = 0.5):
    """Run late fusion inference (ensemble)."""
    from ultralytics.utils.ops import non_max_suppression

    rgb_results = rgb_model.predict(str(rgb_path), imgsz=imgsz, conf=conf, verbose=False)
    thermal_results = thermal_model.predict(str(thermal_path), imgsz=imgsz, conf=conf, verbose=False)

    # Get raw predictions for NMS
    rgb_pred = rgb_model(rgb_path, imgsz=imgsz, conf=conf, verbose=False)
    thermal_pred = thermal_model(thermal_path, imgsz=imgsz, conf=conf, verbose=False)

    # Ensemble: concatenate predictions and NMS
    # This is a simplified version - real implementation would fuse at feature level
    return rgb_results[0], thermal_results[0]


def draw_boxes(image_path: Path, results, output_path: Path, class_names: list = None):
    """Draw bounding boxes on image and save."""
    if class_names is None:
        class_names = ["person"]

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read image: {image_path}")
        return

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{class_names[cls] if cls < len(class_names) else cls}: {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(str(output_path), img)
    print(f"Saved visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on trained models")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--thermal-image", type=str, help="Path to paired thermal image (for fusion)")
    parser.add_argument("--modality", type=str, default="rgb", choices=["rgb", "thermal", "fusion"])
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", type=str, default="0", help="Device (0 for GPU, cpu for CPU)")
    parser.add_argument("--output", type=str, default="result.jpg", help="Output visualization path")
    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    print(f"Loading model: {model_path}")
    model = load_model(model_path)
    model.to(args.device)

    print(f"Running inference on: {image_path}")
    if args.modality in ("rgb", "thermal"):
        results = run_inference_rgb(model, image_path, args.imgsz, args.conf)
        draw_boxes(image_path, results, Path(args.output))

        # Print detections
        print(f"\nDetections: {len(results.boxes)}")
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            print(f"  Person: {conf:.4f} [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

    elif args.modality == "fusion":
        if not args.thermal_image:
            print("Fusion mode requires --thermal-image")
            return
        thermal_path = Path(args.thermal_image)
        if not thermal_path.exists():
            print(f"Thermal image not found: {thermal_path}")
            return

        # For fusion, we'd need both models or a fusion model
        # This is a placeholder
        print("Fusion inference not fully implemented - use trained fusion model directly")
        results = run_inference_rgb(model, image_path, args.imgsz, args.conf)
        draw_boxes(image_path, results, Path(args.output))


if __name__ == "__main__":
    main()