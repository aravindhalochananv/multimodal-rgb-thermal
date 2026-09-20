"""Convert PASCAL VOC XML annotations to YOLO format.

Creates standard YOLO directory structure:
    yolo_dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/

Handles:
- VOC XML → YOLO normalized bbox conversion
- Empty annotations (skipped with logging)
- Invalid bounding boxes (xmin>=xmax, ymin>=ymax) (skipped with logging)
- Annotation/image consistency verification
"""

import csv
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.config import (
    DATA_VISIBLE_TRAIN, DATA_VISIBLE_TEST,
    DATA_INFRARED_TRAIN, DATA_INFRARED_TEST,
    DATA_ANNOTATIONS,
    IMG_WIDTH, IMG_HEIGHT,
    SEED,
)


def parse_voc_annotation_safe(annotation_path: Path) -> Optional[Dict]:
    """Parse a VOC XML annotation file with error handling.

    Returns None if the file cannot be parsed.
    """
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        size = root.find("size")
        if size is None:
            return None
        width = int(size.find("width").text) if size.find("width") is not None else IMG_WIDTH
        height = int(size.find("height").text) if size.find("height") is not None else IMG_HEIGHT

        objects = []
        for obj in root.findall("object"):
            name = obj.find("name").text if obj.find("name") is not None else ""
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            xmin_el = bndbox.find("xmin")
            ymin_el = bndbox.find("ymin")
            xmax_el = bndbox.find("xmax")
            ymax_el = bndbox.find("ymax")
            if any(e is None for e in [xmin_el, ymin_el, xmax_el, ymax_el]):
                continue
            bbox = [
                int(xmin_el.text), int(ymin_el.text),
                int(xmax_el.text), int(ymax_el.text),
            ]
            truncated = int(obj.find("truncated").text) if obj.find("truncated") is not None else 0
            difficult = int(obj.find("difficult").text) if obj.find("difficult") is not None else 0
            objects.append({
                "name": name,
                "bbox": bbox,
                "truncated": truncated,
                "difficult": difficult,
            })

        filename = root.find("filename").text if root.find("filename") is not None else ""

        return {
            "filename": filename,
            "width": width,
            "height": height,
            "objects": objects,
        }
    except Exception:
        return None


def voc_to_yolo_bbox(bbox: List[int], img_w: int, img_h: int) -> Optional[List[float]]:
    """Convert VOC bbox [xmin, ymin, xmax, ymax] to YOLO normalized [x_center, y_center, w, h]."""
    xmin, ymin, xmax, ymax = bbox
    if xmin >= xmax or ymin >= ymax:
        return None
    width = xmax - xmin
    height = ymax - ymin
    return [
        (xmin + width / 2) / img_w,
        (ymin + height / 2) / img_h,
        width / img_w,
        height / img_h,
    ]


def split_files(image_paths: list, val_ratio: float = 0.1, max_val: int = 500, seed: int = 42) -> tuple:
    """Split file list into train and val subsets."""
    random.seed(seed)
    n = len(image_paths)
    val_count = min(max(int(n * val_ratio), 1 if n > 0 else 0), max_val)
    val_set = set(random.sample(range(n), val_count)) if n > 0 else set()
    train = [p for i, p in enumerate(image_paths) if i not in val_set]
    val = [p for i, p in enumerate(image_paths) if i in val_set]
    return train, val


def convert_dataset(
    image_dir: Path,
    annotation_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.1,
    max_val: int = 500,
    seed: int = SEED,
) -> Dict:
    """Convert all VOC annotations in a directory to YOLO format.

    Creates the standard YOLO directory structure with train/val/test splits.
    Returns conversion statistics.
    """
    start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    img_train = output_dir / "images" / "train"
    img_val = output_dir / "images" / "val"
    img_test = output_dir / "images" / "test"
    lbl_train = output_dir / "labels" / "train"
    lbl_val = output_dir / "labels" / "val"
    lbl_test = output_dir / "labels" / "test"

    for d in [img_train, img_val, img_test, lbl_train, lbl_val, lbl_test]:
        d.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions])

    stats = {
        "total_images": len(image_files),
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "total_annotations": 0,
        "converted": 0,
        "skipped_empty": 0,
        "skipped_invalid_bbox": 0,
        "skipped_unmatched_image": 0,
        "errors": 0,
        "test_converted": 0,
        "warnings": [],
    }

    train_paths, val_paths = split_files(image_files, val_ratio=val_ratio, max_val=max_val, seed=seed)

    def process_image(img_path: Path, target_img_dir: Path, target_lbl_dir: Path, is_test: bool = False):
        """Process a single image and its annotation."""
        stem = img_path.stem
        ann_path = annotation_dir / f"{stem}.xml"
        stats["total_annotations"] += 1

        if not ann_path.exists():
            stats["skipped_unmatched_image"] += 1
            return

        ann_data = parse_voc_annotation_safe(ann_path)
        if ann_data is None:
            stats["errors"] += 1
            return

        if not ann_data["objects"]:
            stats["skipped_empty"] += 1
            return

        img_w = ann_data["width"]
        img_h = ann_data["height"]
        has_valid = False

        yolo_lines = []
        for obj in ann_data["objects"]:
            yolo_bbox = voc_to_yolo_bbox(obj["bbox"], img_w, img_h)
            if yolo_bbox is None:
                stats["skipped_invalid_bbox"] += 1
                continue
            yolo_lines.append(f"0 {' '.join(f'{v:.6f}' for v in yolo_bbox)}")
            has_valid = True

        if has_valid:
            shutil.copy2(img_path, target_img_dir / img_path.name)
            label_path = target_lbl_dir / f"{stem}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines) + "\n")
            if is_test:
                stats["test_converted"] += 1
            else:
                stats["converted"] += 1
        else:
            stats["skipped_invalid_bbox"] += 1

    for f in train_paths:
        process_image(f, img_train, lbl_train)
    stats["train_images"] = len(list(img_train.iterdir()))

    for f in val_paths:
        process_image(f, img_val, lbl_val)
    stats["val_images"] = len(list(img_val.iterdir()))

    test_image_dir = (
        DATA_VISIBLE_TEST if str(image_dir) == str(DATA_VISIBLE_TRAIN)
        else DATA_INFRARED_TEST
    )
    test_files = sorted([f for f in test_image_dir.iterdir() if f.suffix.lower() in image_extensions])
    for f in test_files:
        process_image(f, img_test, lbl_test, is_test=True)
    stats["test_images"] = len(list(img_test.iterdir()))

    stats["elapsed"] = time.time() - start
    return stats


def create_yaml(
    dataset_dir: Path,
    num_classes: int = 1,
    class_names: list = None,
) -> str:
    """Create a YOLO dataset YAML configuration file."""
    if class_names is None:
        class_names = ["person"]

    yaml_content = f"""path: {dataset_dir}
train: images/train
val: images/val
test: images/test

nc: {num_classes}
names: {class_names}
"""
    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    return yaml_path


def verify_conversion(dataset_dir: Path) -> Dict:
    """Verify YOLO conversion correctness."""
    stats = {
        "train_images": 0, "train_labels": 0, "train_orphan_images": 0, "train_orphan_labels": 0,
        "val_images": 0, "val_labels": 0, "val_orphan_images": 0, "val_orphan_labels": 0,
        "test_images": 0, "test_labels": 0, "test_orphan_images": 0, "test_orphan_labels": 0,
        "total_objects": 0, "empty_labels": 0,
    }

    for split in ["train", "val", "test"]:
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        if not images_dir.exists() or not labels_dir.exists():
            continue

        image_files = set(f.stem for f in images_dir.iterdir() if f.is_file())
        label_files = set(f.stem for f in labels_dir.iterdir() if f.is_file())

        key_prefix = split
        stats[f"{key_prefix}_images"] = len(image_files)
        stats[f"{key_prefix}_labels"] = len(label_files)
        stats[f"{key_prefix}_orphan_images"] = len(image_files - label_files)
        stats[f"{key_prefix}_orphan_labels"] = len(label_files - image_files)

        for label_file in labels_dir.iterdir():
            if label_file.suffix != ".txt":
                continue
            content = label_file.read_text().strip()
            if not content:
                stats["empty_labels"] += 1
            else:
                for line in content.split("\n"):
                    if line.strip():
                        stats["total_objects"] += 1

    return stats
