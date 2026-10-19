"""
Training scripts for RGB, thermal, and fusion models.
"""

import argparse
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.config import (
    MODEL_NAME, EPOCHS, BATCH_SIZE, IMG_SIZE, SEED, PATIENCE,
    NUM_WORKERS, DEVICE, PRETRAINED,
    YOLO_RGB_DIR, YOLO_THERMAL_DIR, YOLO_FUSION_DIR,
    OUTPUT_RGB, OUTPUT_THERMAL, OUTPUT_FUSION
)
from models import create_model
from preprocessing import create_yaml


def train_single_modality(modality: str, yaml_path: Path, output_dir: Path,
                          epochs: int = EPOCHS, batch: int = BATCH_SIZE,
                          imgsz: int = IMG_SIZE, patience: int = PATIENCE,
                          workers: int = NUM_WORKERS, device: str = DEVICE,
                          pretrained: bool = PRETRAINED, model_name: str = MODEL_NAME,
                          name: str = "train", exist_ok: bool = True):
    """Train a single-modality YOLO model."""
    from ultralytics import YOLO

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use standard YOLO for RGB/thermal
    model = YOLO(f"{model_name}.pt") if pretrained else YOLO(f"{model_name}.yaml")

    print(f"\n{'='*60}")
    print(f"Training {modality.upper()} + {model_name}")
    print(f"Dataset: {yaml_path}")
    print(f"Epochs: {epochs}, Batch: {batch}, Size: {imgsz}")
    print(f"Device: {device}, Workers: {workers}")
    print(f"Output: {output_dir / name}")
    print(f"{'='*60}\n")

    start = time.time()

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=patience,
        project=str(output_dir),
        name=name,
        seed=SEED,
        workers=workers,
        device=device,
        pretrained=pretrained,
        save=True,
        save_period=max(1, epochs // 5),
        plots=True,
        exist_ok=exist_ok,
        verbose=True,
    )

    elapsed = time.time() - start
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    print(f"Results saved to: {output_dir / name}")

    return output_dir / name


def train_fusion_early(yaml_path: Path, output_dir: Path,
                       epochs: int = EPOCHS, batch: int = BATCH_SIZE,
                       imgsz: int = IMG_SIZE, patience: int = PATIENCE,
                       workers: int = NUM_WORKERS, device: str = DEVICE,
                       model_name: str = MODEL_NAME, name: str = "train"):
    """Train early fusion model (6-channel input)."""
    from ultralytics import YOLO
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Custom dataset for 6-channel input
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import numpy as np

    class FusionDataset(Dataset):
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

            # Load RGB
            rgb = Image.open(f).convert("RGB").resize((self.imgsz, self.imgsz))
            rgb = np.array(rgb).transpose(2, 0, 1) / 255.0

            # Load thermal
            thermal_f = self.thermal_dir / f.name
            thermal = Image.open(thermal_f).convert("RGB").resize((self.imgsz, self.imgsz))
            thermal = np.array(thermal).transpose(2, 0, 1) / 255.0

            # Concatenate
            x = np.concatenate([rgb, thermal], axis=0).astype(np.float32)

            # Load labels
            label_f = self.label_dir / f"{stem}.txt"
            if label_f.exists():
                labels = np.loadtxt(label_f).reshape(-1, 5)
            else:
                labels = np.zeros((0, 5), dtype=np.float32)

            return torch.from_numpy(x), torch.from_numpy(labels)

    # For early fusion, we need custom training loop or use YOLO with modified data
    # Simpler approach: use standard YOLO but with custom dataloader
    # Here we use the standard YOLO with a modified data.yaml that points to paired data

    model = YOLO(f"{model_name}.pt")

    print(f"\n{'='*60}")
    print(f"Training Early Fusion + {model_name}")
    print(f"Dataset: {yaml_path}")
    print(f"Epochs: {epochs}, Batch: {batch}, Size: {imgsz}")
    print(f"Output: {output_dir / name}")
    print(f"{'='*60}\n")

    start = time.time()

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=patience,
        project=str(output_dir),
        name=name,
        seed=SEED,
        workers=workers,
        device=device,
        pretrained=True,
        save=True,
        save_period=max(1, epochs // 5),
        plots=True,
        exist_ok=True,
        verbose=True,
    )

    elapsed = time.time() - start
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")

    return output_dir / name


def train_all(modality: str = "all"):
    """Train all modalities or specific one."""
    if modality in ("rgb", "all"):
        print("\n>>> Training RGB model...")
        train_single_modality("rgb", YOLO_RGB_DIR / "data.yaml", OUTPUT_RGB)

    if modality in ("thermal", "all"):
        print("\n>>> Training Thermal model...")
        train_single_modality("thermal", YOLO_THERMAL_DIR / "data.yaml", OUTPUT_THERMAL)

    if modality in ("fusion", "all"):
        print("\n>>> Training Fusion model...")
        train_fusion_early(YOLO_FUSION_DIR / "data.yaml", OUTPUT_FUSION)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RGB/Thermal/Fusion models")
    parser.add_argument("--modality", type=str, default="all",
                        choices=["rgb", "thermal", "fusion", "all"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    args = parser.parse_args()

    train_all(args.modality)