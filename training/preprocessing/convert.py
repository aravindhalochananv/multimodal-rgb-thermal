"""
Dataset preprocessing for LLVIP: VOC XML to YOLO format conversion.
Handles RGB, thermal, and fusion datasets with train/val/test splits.
"""

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json


@dataclass
class ConversionStats:
    total_images: int = 0
    train_images: int = 0
    val_images: int = 0
    test_images: int = 0
    total_annotations: int = 0
    converted: int = 0
    test_converted: int = 0
    skipped_empty: int = 0
    skipped_invalid_bbox: int = 0
    skipped_unmatched: int = 0
    errors: int = 0


def parse_voc_annotation(annotation_path: Path) -> Optional[Dict]:
    """Parse PASCAL VOC XML annotation with error handling."""
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        size = root.find("size")
        width = int(size.find("width").text) if size is not None and size.find("width") is not None else 1280
        height = int(size.find("height").text) if size is not None and size.find("height") is not None else 1024

        objects = []
        for obj in root.findall("object"):
            name_el = obj.find("name")
            bndbox = obj.find("bndbox")
            if name_el is None or bndbox is None:
                continue
            els = [bndbox.find(t) for t in ("xmin", "ymin", "xmax", "ymax")]
            if any(e is None for e in els):
                continue
            objects.append({
                "name": name_el.text,
                "bbox": [int(e.text) for e in els]
            })

        return {"width": width, "height": height, "objects": objects}
    except Exception:
        return None


def voc_to_yolo_bbox(bbox: List[int], img_w: int, img_h: int) -> Optional[List[float]]:
    """Convert VOC [xmin, ymin, xmax, ymax] to YOLO [xc, yc, w, h] normalized."""
    xmin, ymin, xmax, ymax = bbox
    if xmin >= xmax or ymin >= ymax:
        return None
    w, h = xmax - xmin, ymax - ymin
    return [
        (xmin + w / 2) / img_w,
        (ymin + h / 2) / img_h,
        w / img_w,
        h / img_h
    ]


def split_files(image_paths: List[Path], val_ratio: float = 0.1,
                max_val: int = 500, seed: int = 42) -> Tuple[List[Path], List[Path]]:
    """Split into train/val with fixed seed for reproducibility."""
    random.seed(seed)
    n = len(image_paths)
    val_count = min(max(int(n * val_ratio), 1 if n > 0 else 0), max_val)
    val_set = set(random.sample(range(n), val_count)) if n > 0 else set()
    train = [p for i, p in enumerate(image_paths) if i not in val_set]
    val = [p for i, p in enumerate(image_paths) if i in val_set]
    return train, val


def process_image(img_path: Path, annotation_dir: Path,
                  img_target_dir: Path, lbl_target_dir: Path,
                  empty_annotations: set, zero_width_boxes: set,
                  is_test: bool = False,
                  stats: Optional[ConversionStats] = None) -> bool:
    """Process single image + annotation. Returns True if converted."""
    stem = img_path.stem

    if stats:
        stats.total_annotations += 1

    # Check for known problematic files
    if stem in empty_annotations:
        if stats:
            stats.skipped_empty += 1
        return False

    ann_path = annotation_dir / f"{stem}.xml"
    if not ann_path.exists():
        if stats:
            stats.skipped_unmatched += 1
        return False

    ann = parse_voc_annotation(ann_path)
    if ann is None:
        if stats:
            stats.errors += 1
        return False

    if not ann["objects"]:
        if stats:
            stats.skipped_empty += 1
        return False

    lines, valid = [], False
    for obj in ann["objects"]:
        if stem in zero_width_boxes:
            if stats:
                stats.skipped_invalid_bbox += 1
            continue
        yolo_bbox = voc_to_yolo_bbox(obj["bbox"], ann["width"], ann["height"])
        if yolo_bbox is None:
            if stats:
                stats.skipped_invalid_bbox += 1
            continue
        lines.append("0 " + " ".join(f"{v:.6f}" for v in yolo_bbox))
        valid = True

    if not valid:
        if stats:
            stats.skipped_invalid_bbox += 1
        return False

    # Copy image and write label
    shutil.copy2(img_path, img_target_dir / img_path.name)
    (lbl_target_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    if stats:
        if is_test:
            stats.test_converted += 1
        else:
            stats.converted += 1
    return True


def convert_modality(image_dir: Path, test_image_dir: Path, annotation_dir: Path,
                     output_dir: Path, val_ratio: float = 0.1,
                     max_val: int = 500, seed: int = 42,
                     empty_annotations: Optional[set] = None,
                     zero_width_boxes: Optional[set] = None) -> ConversionStats:
    """Convert a single modality (RGB or thermal) to YOLO format."""
    if empty_annotations is None:
        empty_annotations = set()
    if zero_width_boxes is None:
        zero_width_boxes = set()

    output_dir = Path(output_dir)
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = ConversionStats()

    # Get training images
    image_files = sorted(f for f in image_dir.iterdir()
                         if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    stats.total_images = len(image_files)

    train_paths, val_paths = split_files(image_files, val_ratio, max_val, seed)

    # Process train
    for f in train_paths:
        process_image(f, annotation_dir,
                      output_dir / "images" / "train",
                      output_dir / "labels" / "train",
                      empty_annotations, zero_width_boxes,
                      is_test=False, stats=stats)

    # Process val
    for f in val_paths:
        process_image(f, annotation_dir,
                      output_dir / "images" / "val",
                      output_dir / "labels" / "val",
                      empty_annotations, zero_width_boxes,
                      is_test=False, stats=stats)

    # Process test
    test_files = sorted(f for f in test_image_dir.iterdir()
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    for f in test_files:
        process_image(f, annotation_dir,
                      output_dir / "images" / "test",
                      output_dir / "labels" / "test",
                      empty_annotations, zero_width_boxes,
                      is_test=True, stats=stats)

    stats.train_images = len(list((output_dir / "images" / "train").iterdir()))
    stats.val_images = len(list((output_dir / "images" / "val").iterdir()))
    stats.test_images = len(list((output_dir / "images" / "test").iterdir()))

    return stats


def create_yaml(dataset_dir: Path, num_classes: int = 1,
                class_names: Optional[List[str]] = None) -> Path:
    """Create YOLO data.yaml configuration file."""
    if class_names is None:
        class_names = ["person"]

    content = (f"path: {dataset_dir.resolve()}\n"
               f"train: images/train\n"
               f"val: images/val\n"
               f"test: images/test\n\n"
               f"nc: {num_classes}\n"
               f"names: {class_names}\n")
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(content)
    return yaml_path


def verify_conversion(dataset_dir: Path) -> Dict:
    """Verify YOLO conversion correctness."""
    dataset_dir = Path(dataset_dir)
    stats = {}
    for split in ("train", "val", "test"):
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        imgs = {f.stem for f in img_dir.iterdir()} if img_dir.exists() else set()
        lbls = {f.stem for f in lbl_dir.iterdir()} if lbl_dir.exists() else set()
        stats[f"{split}_images"] = len(imgs)
        stats[f"{split}_labels"] = len(lbls)
        stats[f"{split}_orphan_images"] = len(imgs - lbls)
        stats[f"{split}_orphan_labels"] = len(lbls - imgs)
    return stats


def convert_all_modalities(config) -> Dict:
    """Convert RGB, thermal, and fusion datasets."""
    results = {}

    # RGB conversion
    print("\n=== Converting RGB dataset ===")
    stats = convert_modality(
        image_dir=config.VISIBLE_TRAIN,
        test_image_dir=config.VISIBLE_TEST,
        annotation_dir=config.ANNOTATIONS,
        output_dir=config.YOLO_RGB_DIR,
        val_ratio=config.VAL_RATIO,
        max_val=config.MAX_VAL_IMAGES,
        seed=config.SEED,
        empty_annotations=config.EMPTY_ANNOTATIONS,
        zero_width_boxes=config.ZERO_WIDTH_BOXES
    )
    create_yaml(config.YOLO_RGB_DIR, config.NUM_CLASSES, config.CLASS_NAMES)
    results["rgb"] = {"conversion": asdict(stats), "verification": verify_conversion(config.YOLO_RGB_DIR)}
    print(f"RGB: {stats.train_images} train, {stats.val_images} val, {stats.test_images} test")

    # Thermal conversion
    print("\n=== Converting Thermal dataset ===")
    stats = convert_modality(
        image_dir=config.INFRARED_TRAIN,
        test_image_dir=config.INFRARED_TEST,
        annotation_dir=config.ANNOTATIONS,
        output_dir=config.YOLO_THERMAL_DIR,
        val_ratio=config.VAL_RATIO,
        max_val=config.MAX_VAL_IMAGES,
        seed=config.SEED,
        empty_annotations=config.EMPTY_ANNOTATIONS,
        zero_width_boxes=config.ZERO_WIDTH_BOXES
    )
    create_yaml(config.YOLO_THERMAL_DIR, config.NUM_CLASSES, config.CLASS_NAMES)
    results["thermal"] = {"conversion": asdict(stats), "verification": verify_conversion(config.YOLO_THERMAL_DIR)}
    print(f"Thermal: {stats.train_images} train, {stats.val_images} val, {stats.test_images} test")

    # Fusion: early fusion dataset (paired RGB+thermal as 6-channel)
    print("\n=== Preparing Fusion dataset (early fusion) ===")
    # For early fusion, we create a dataset where each sample is a pair
    # This requires custom dataloader - here we prepare the paired structure
    fusion_prepare_paired(config)
    results["fusion"] = {"status": "paired_structure_created"}

    return results


def fusion_prepare_paired(config) -> None:
    """Prepare paired RGB-thermal structure for fusion training."""
    # Create directory structure
    for split in ("train", "val", "test"):
        (config.YOLO_FUSION_DIR / "images" / split / "rgb").mkdir(parents=True, exist_ok=True)
        (config.YOLO_FUSION_DIR / "images" / split / "thermal").mkdir(parents=True, exist_ok=True)
        (config.YOLO_FUSION_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Get paired stems from RGB train
    rgb_train_stems = {f.stem for f in config.VISIBLE_TRAIN.iterdir()
                       if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}}
    thermal_train_stems = {f.stem for f in config.INFRARED_TRAIN.iterdir()
                           if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}}
    paired_stems = sorted(rgb_train_stems & thermal_train_stems)

    # Split paired data
    train_stems, val_stems = split_files([Path(s) for s in paired_stems],
                                         config.VAL_RATIO, config.MAX_VAL_IMAGES, config.SEED)
    train_stems = [p.name for p in train_stems]
    val_stems = [p.name for p in val_stems]

    # Test stems
    test_stems = {f.stem for f in config.VISIBLE_TEST.iterdir()
                  if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}}

    def copy_pair(stem: str, split: str):
        shutil.copy2(config.VISIBLE_TRAIN / f"{stem}.jpg" if split != "test" else config.VISIBLE_TEST / f"{stem}.jpg",
                     config.YOLO_FUSION_DIR / "images" / split / "rgb" / f"{stem}.jpg")
        shutil.copy2(config.INFRARED_TRAIN / f"{stem}.jpg" if split != "test" else config.INFRARED_TEST / f"{stem}.jpg",
                     config.YOLO_FUSION_DIR / "images" / split / "thermal" / f"{stem}.jpg")
        # Copy label (same for both)
        shutil.copy2(config.YOLO_RGB_DIR / "labels" / split / f"{stem}.txt",
                     config.YOLO_FUSION_DIR / "labels" / split / f"{stem}.txt")

    for stem in train_stems:
        copy_pair(stem, "train")
    for stem in val_stems:
        copy_pair(stem, "val")
    for stem in test_stems:
        copy_pair(stem, "test")

    # Create fusion yaml
    create_yaml(config.YOLO_FUSION_DIR, config.NUM_CLASSES, config.CLASS_NAMES)
    print(f"Fusion: {len(train_stems)} train, {len(val_stems)} val, {len(test_stems)} test pairs")


if __name__ == "__main__":
    # Test run
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from configs.config import *

    print("Testing conversion...")
    results = convert_all_modalities(sys.modules['configs.config'])
    print(json.dumps(results, indent=2))