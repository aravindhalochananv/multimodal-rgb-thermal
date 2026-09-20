"""Preprocessing pipeline for RGB-thermal image pairs.

Includes: loading, resizing, normalization, augmentation, and annotation
conversion to detection format.
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET


def resize_image(
    image: Image.Image,
    size: int | Tuple[int, int],
    interpolation=Image.BILINEAR,
) -> Image.Image:
    if isinstance(size, int):
        return image.resize((size, size), interpolation=interpolation)
    return image.resize(size, interpolation=interpolation)


def normalize_image(
    image: Image.Image,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> np.ndarray:
    arr = np.array(image, dtype=np.float32) / 255.0
    if mean is not None and std is not None:
        mean = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
        std = np.array(std, dtype=np.float32).reshape(1, 1, -1)
        arr = (arr - mean) / std
    return arr


def resize_annotation_bboxes(
    bboxes: List[List[float]],
    original_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> List[List[float]]:
    orig_w, orig_h = original_size
    target_w, target_h = target_size
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    resized = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        resized.append([xmin * scale_x, ymin * scale_y, xmax * scale_x, ymax * scale_y])
    return resized


def parse_voc_annotation(annotation_path: Path) -> Dict:
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    depth = int(size.find("depth").text) if size.find("depth") is not None else 3

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        bbox = [
            int(bndbox.find("xmin").text),
            int(bndbox.find("ymin").text),
            int(bndbox.find("xmax").text),
            int(bndbox.find("ymax").text),
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
        "depth": depth,
        "objects": objects,
    }


def annotations_to_detection_format(
    annotation_path: Path,
    target_size: Optional[Tuple[int, int]] = None,
) -> Dict:
    ann = parse_voc_annotation(annotation_path)
    detection_format = {
        "image_id": Path(annotation_path).stem,
        "filename": ann["filename"],
        "image_width": ann["width"],
        "image_height": ann["height"],
        "annotations": [],
    }

    for obj in ann["objects"]:
        bbox = obj["bbox"]
        if target_size is not None:
            bbox = resize_annotation_bboxes([bbox], (ann["width"], ann["height"]), target_size)[0]

        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin

        detection_format["annotations"].append({
            "class": obj["name"],
            "bbox": [xmin, ymin, xmax, ymax],
            "bbox_width": width,
            "bbox_height": height,
            "area": width * height,
            "truncated": obj["truncated"],
            "difficult": obj["difficult"],
        })

    return detection_format
