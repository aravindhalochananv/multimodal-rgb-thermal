"""
Model export, packaging, and download utilities.
Exports to ONNX, TorchScript, creates deployment packages.
"""

import shutil
import zipfile
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import torch

from ultralytics import YOLO


def export_model(model_path: Path, output_dir: Path,
                 formats: List[str] = None,
                 imgsz: int = 640,
                 simplify: bool = True,
                 opset: int = 12) -> Dict[str, Path]:
    """Export YOLO model to multiple formats."""
    if formats is None:
        formats = ["onnx", "torchscript"]

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    results = {}

    for fmt in formats:
        print(f"Exporting to {fmt.upper()}...")
        try:
            if fmt == "onnx":
                exported = model.export(
                    format="onnx",
                    imgsz=imgsz,
                    simplify=simplify,
                    opset=opset,
                    dynamic=True,
                )
            elif fmt == "torchscript":
                exported = model.export(
                    format="torchscript",
                    imgsz=imgsz,
                    optimize=True,
                )
            elif fmt == "tflite":
                exported = model.export(
                    format="tflite",
                    imgsz=imgsz,
                    int8=False,
                )
            elif fmt == "engine":
                exported = model.export(
                    format="engine",
                    imgsz=imgsz,
                    half=True,
                    device=0,
                )
            else:
                print(f"  Unknown format: {fmt}")
                continue

            # Copy to output directory with clean name
            model_name = model_path.stem
            dst = output_dir / f"{model_name}.{fmt}"
            if Path(exported).exists():
                shutil.copy2(exported, dst)
                results[fmt] = dst
                print(f"  Saved: {dst}")
            else:
                print(f"  Export path not found: {exported}")

        except Exception as e:
            print(f"  ERROR exporting to {fmt}: {e}")
            results[fmt] = None

    return results


def create_deployment_package(model_dir: Path, output_path: Path,
                              include_formats: List[str] = None) -> Path:
    """Create a deployment zip with model, config, and inference script."""
    model_dir = Path(model_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if include_formats is None:
        include_formats = ["pt", "onnx", "torchscript"]

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add PyTorch weights
        weights_dir = model_dir / "weights"
        if weights_dir.exists():
            for f in weights_dir.glob("*.pt"):
                if f.suffix[1:] in include_formats:
                    zf.write(f, f"weights/{f.name}")

        # Add exported formats
        for ext in include_formats:
            for f in model_dir.parent.glob(f"*.{ext}"):
                zf.write(f, f"exports/{f.name}")

        # Add data.yaml
        yaml_path = model_dir.parent / "yolo_dataset" / "data.yaml"
        if yaml_path.exists():
            zf.write(yaml_path, "data.yaml")

        # Add results.csv if exists
        results_csv = model_dir / "results.csv"
        if results_csv.exists():
            zf.write(results_csv, "training_results.csv")

        # Add test results
        test_results = model_dir.parent / "test_results.json"
        if test_results.exists():
            zf.write(test_results, "test_results.json")

        # Add inference script
        inference_script = '''"""
Inference script for deployed model.
Usage: python inference.py --model weights/best.pt --image test.jpg
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model weights (.pt, .onnx, .torchscript)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", type=str, default="0", help="Device (0 for GPU, cpu for CPU)")
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.predict(args.image, imgsz=args.imgsz, conf=args.conf, device=args.device)

    for r in results:
        print(f"Image: {args.image}")
        print(f"Boxes: {len(r.boxes)}")
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            print(f"  Class {cls}: {conf:.4f} [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")

        # Save visualization
        r.save(filename="result.jpg")
        print("Result saved to result.jpg")

if __name__ == "__main__":
    main()
'''
        zf.writestr("inference.py", inference_script)

        # Add README
        readme = f"""# Deployment Package

Created: {datetime.now().isoformat()}
Model: {model_dir.name}

## Contents
- weights/     - PyTorch weights (.pt)
- exports/     - Exported formats (ONNX, TorchScript, etc.)
- data.yaml    - Dataset configuration
- training_results.csv - Training metrics per epoch
- test_results.json    - Test set evaluation metrics
- inference.py - Inference script

## Quick Start
```bash
pip install ultralytics opencv-python
python inference.py --model weights/best.pt --image your_image.jpg
```

## Using ONNX
```bash
pip install onnxruntime
python -c "
import onnxruntime as ort
session = ort.InferenceSession('exports/model.onnx')
# ... run inference
"
```
"""
        zf.writestr("README.md", readme)

    print(f"Deployment package created: {output_path}")
    return output_path


def download_from_colab(file_path: Path) -> None:
    """Trigger file download in Google Colab."""
    try:
        from google.colab import files
        files.download(str(file_path))
        print(f"Download started for: {file_path}")
    except ImportError:
        print("Not in Colab environment. File saved locally at:", file_path)


def package_all_outputs(work_dir: Path, output_zip: Path) -> Path:
    """Package all training outputs into a single zip."""
    work_dir = Path(work_dir)
    output_zip = Path(output_path)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(work_dir):
            for f in files:
                if f.endswith(('.pt', '.onnx', '.torchscript', '.yaml', '.csv', '.json', '.png', '.jpg')):
                    src = Path(root) / f
                    rel = src.relative_to(work_dir)
                    zf.write(src, rel)

    print(f"All outputs packaged: {output_zip}")
    return output_zip


def save_training_artifacts(model_dir: Path, config: Dict) -> None:
    """Save all training artifacts with metadata."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = model_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Save git info if available
    try:
        import subprocess
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        git_branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
        with open(model_dir / "git_info.txt", "w") as f:
            f.write(f"commit: {git_hash}\\nbranch: {git_branch}\\n")
    except Exception:
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export and package models")
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--formats", nargs="+", default=["onnx", "torchscript"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--package", action="store_true", help="Create deployment package")
    parser.add_argument("--download", action="store_true", help="Trigger Colab download")
    args = parser.parse_args()

    # Export
    results = export_model(Path(args.model), Path(args.output), args.formats, args.imgsz)

    # Package
    if args.package:
        pkg_path = Path(args.output) / f"{Path(args.model).stem}_deployment.zip"
        create_deployment_package(Path(args.model).parent, pkg_path)
        if args.download:
            download_from_colab(pkg_path)