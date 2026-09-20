"""Generic YOLO training script for single-modality detection.

Usage:
    python -m src.training.train_yolo --modality rgb --model yolov8n --epochs 3 --batch-size 4 --img-size 320
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.config import (
    DATA_VISIBLE_TRAIN, DATA_VISIBLE_TEST,
    DATA_INFRARED_TRAIN, DATA_INFRARED_TEST,
    DATA_ANNOTATIONS,
    OUTPUT_RGB, OUTPUT_THERMAL,
    SEED,
)
from src.data.voc_to_yolo import convert_dataset, create_yaml, verify_conversion


def run(
    modality: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    img_size: int,
    patience: int = 10,
    force_rebuild: bool = False,
):
    """Run a complete YOLO training experiment."""
    from ultralytics import YOLO

    if modality == "rgb":
        img_dir_train = DATA_VISIBLE_TRAIN
        img_dir_test = DATA_VISIBLE_TEST
        output_base = OUTPUT_RGB / model_name
    else:
        img_dir_train = DATA_INFRARED_TRAIN
        img_dir_test = DATA_INFRARED_TEST
        output_base = OUTPUT_THERMAL / model_name

    yolo_dir = output_base / "yolo_dataset"
    yaml_path = yolo_dir / "data.yaml"

    if yolo_dir.exists() and not force_rebuild:
        print(f"Dataset exists at {yolo_dir}. Use --force-rebuild.")
    else:
        print(f"\nConverting {modality.upper()} dataset to YOLO format...")
        stats = convert_dataset(
            image_dir=img_dir_train,
            annotation_dir=DATA_ANNOTATIONS,
            output_dir=yolo_dir,
        )
        print(f"Conversion stats: {stats}")

        create_yaml(yolo_dir)
        verify = verify_conversion(yolo_dir)
        print(f"Verification: {verify}")

    output_dir = output_base / "train"
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_name == "yolov8n":
        model = YOLO("yolov8n.pt")
    elif model_name == "yolov8s":
        model = YOLO("yolov8s.pt")
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    print(f"\nTraining {modality.upper()} + {model_name}")
    print(f"Dataset: {yolo_dir}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, Size: {img_size}")

    model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=patience,
        project=str(output_base),
        name="train",
        seed=SEED,
        workers=0,
        device="cpu",
        pretrained=True,
        save=True,
        save_period=epochs if epochs <= 5 else max(1, epochs // 5),
        plots=True,
        verbose=True,
        exist_ok=True,
    )

    print(f"\nTraining complete. Results in: {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, required=True, choices=["rgb", "thermal"])
    parser.add_argument("--model", type=str, required=True, choices=["yolov8n", "yolov8s"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--force-rebuild", action="store_true")

    args = parser.parse_args()

    run(
        modality=args.modality,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        patience=args.patience,
        force_rebuild=args.force_rebuild,
    )
