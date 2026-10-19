"""
Inference module for Multimodal RGB-Thermal Human Detection.
Supports single-modality prediction (RGB or Thermal) and Multimodal Fusion prediction,
as well as generating comparative side-by-side visual detections.

Usage Examples:
    # Predict with RGB model:
    python -m src.inference.predict --modality rgb --input path/to/image.jpg

    # Predict with Thermal model:
    python -m src.inference.predict --modality thermal --input path/to/thermal.jpg

    # Predict with Multimodal Fusion:
    python -m src.inference.predict --modality multimodal --rgb path/to/rgb.jpg --thermal path/to/thermal.jpg

    # Generate 3-way comparative panel (RGB vs Thermal vs Fusion):
    python -m src.inference.predict --compare --rgb path/to/rgb.jpg --thermal path/to/thermal.jpg --output-dir outputs/predictions
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Union, List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.config import OUTPUT_RGB, OUTPUT_THERMAL, OUTPUT_MULTIMODAL, MODEL_NAME
from src.models.fusion_model import PixelFusionYOLO


def load_model(modality: str, weights_path: Optional[Union[str, Path]] = None, model_name: str = MODEL_NAME) -> YOLO:
    """Load YOLO model for specified modality."""
    if weights_path is not None:
        p = Path(weights_path)
    else:
        if modality == "rgb":
            p = OUTPUT_RGB / model_name / "train" / "weights" / "best.pt"
        elif modality == "thermal":
            p = OUTPUT_THERMAL / model_name / "train" / "weights" / "best.pt"
        else:
            p = OUTPUT_MULTIMODAL / model_name / "train" / "weights" / "best.pt"

    if not p.exists():
        fallback = repo_root / "yolov8n.pt"
        print(f"Warning: Model weights not found at {p}. Falling back to {fallback}")
        p = fallback

    return YOLO(str(p))


def predict_single(
    image_path: Union[str, Path],
    modality: str = "rgb",
    weights_path: Optional[Union[str, Path]] = None,
    imgsz: int = 320,
    conf: float = 0.25,
    iou: float = 0.5,
    save_path: Optional[Union[str, Path]] = None,
):
    """Run prediction on a single image and optionally save result."""
    model = load_model(modality, weights_path)
    results = model.predict(str(image_path), imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    result = results[0]

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(save_path))
        print(f"Prediction saved to: {save_path}")

    return result


def predict_multimodal(
    rgb_path: Union[str, Path],
    thermal_path: Union[str, Path],
    weights_path: Optional[Union[str, Path]] = None,
    alpha: float = 0.5,
    imgsz: int = 320,
    conf: float = 0.25,
    iou: float = 0.5,
    save_path: Optional[Union[str, Path]] = None,
):
    """Run prediction using pixel-level multimodal fusion."""
    fusion_model = PixelFusionYOLO(weights_path=weights_path, alpha=alpha)
    results = fusion_model.predict(
        rgb_input=rgb_path,
        thermal_input=thermal_path,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
    )
    result = results[0]

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(save_path))
        print(f"Multimodal prediction saved to: {save_path}")

    return result


def generate_comparison_panel(
    rgb_path: Union[str, Path],
    thermal_path: Union[str, Path],
    output_path: Union[str, Path],
    conf: float = 0.25,
    imgsz: int = 320,
) -> np.ndarray:
    """
    Run RGB, Thermal, and Fusion models on paired images and concatenate results side-by-side.
    """
    m_rgb = load_model("rgb")
    m_th = load_model("thermal")
    m_fu = PixelFusionYOLO()

    # Predictions
    res_rgb = m_rgb.predict(str(rgb_path), imgsz=imgsz, conf=conf, verbose=False)[0]
    res_th = m_th.predict(str(thermal_path), imgsz=imgsz, conf=conf, verbose=False)[0]
    res_fu = m_fu.predict(rgb_path, thermal_path, imgsz=imgsz, conf=conf, verbose=False)[0]

    # Plot annotated results
    plot_rgb = res_rgb.plot()
    plot_th = res_th.plot()
    plot_fu = res_fu.plot()

    # Ensure identical heights
    h = min(plot_rgb.shape[0], plot_th.shape[0], plot_fu.shape[0])
    w = int(plot_rgb.shape[1] * (h / plot_rgb.shape[0]))
    p_rgb = cv2.resize(plot_rgb, (w, h))
    p_th = cv2.resize(plot_th, (w, h))
    p_fu = cv2.resize(plot_fu, (w, h))

    # Add titles on top of images
    def add_header(img, text, color=(255, 255, 255), bg=(40, 40, 40)):
        banner = np.zeros((40, img.shape[1], 3), dtype=np.uint8)
        banner[:] = bg
        cv2.putText(banner, text, (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return np.vstack([banner, img])

    h_rgb = add_header(p_rgb, f"RGB-Only ({len(res_rgb.boxes)} detected)", bg=(70, 30, 20))
    h_th = add_header(p_th, f"Thermal-Only ({len(res_th.boxes)} detected)", bg=(20, 40, 70))
    h_fu = add_header(p_fu, f"50:50 Fusion ({len(res_fu.boxes)} detected)", bg=(50, 20, 60))

    panel = np.hstack([h_rgb, h_th, h_fu])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), panel)
    print(f"Comparison panel saved to: {output_path}")
    return panel


def main():
    parser = argparse.ArgumentParser(description="Multimodal Human Detection Inference")
    parser.add_argument("--modality", choices=["rgb", "thermal", "multimodal"], default="rgb")
    parser.add_argument("--input", type=str, help="Input image path for single modality")
    parser.add_argument("--rgb", type=str, help="RGB image path for multimodal or comparison")
    parser.add_argument("--thermal", type=str, help="Thermal image path for multimodal or comparison")
    parser.add_argument("--compare", action="store_true", help="Generate 3-way comparative panel")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference image size")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions", help="Output directory")

    args = parser.parse_args()

    if args.compare:
        if not args.rgb or not args.thermal:
            print("Error: --compare requires both --rgb and --thermal inputs.")
            sys.exit(1)
        stem = Path(args.rgb).stem
        out_p = Path(args.output) if args.output else Path(args.output_dir) / f"{stem}_comparison.jpg"
        generate_comparison_panel(args.rgb, args.thermal, out_p, conf=args.conf, imgsz=args.imgsz)
    elif args.modality == "multimodal":
        if not args.rgb or not args.thermal:
            print("Error: multimodal modality requires both --rgb and --thermal inputs.")
            sys.exit(1)
        stem = Path(args.rgb).stem
        out_p = Path(args.output) if args.output else Path(args.output_dir) / f"{stem}_fused_pred.jpg"
        predict_multimodal(args.rgb, args.thermal, conf=args.conf, imgsz=args.imgsz, save_path=out_p)
    else:
        in_p = args.input or (args.rgb if args.modality == "rgb" else args.thermal)
        if not in_p:
            print("Error: Please provide --input (or --rgb/--thermal).")
            sys.exit(1)
        stem = Path(in_p).stem
        out_p = Path(args.output) if args.output else Path(args.output_dir) / f"{stem}_{args.modality}_pred.jpg"
        predict_single(in_p, modality=args.modality, conf=args.conf, imgsz=args.imgsz, save_path=out_p)


if __name__ == "__main__":
    main()
