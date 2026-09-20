"""LLVIP dataset loading utilities.

Loads paired RGB/visible and thermal/infrared images with error handling
for missing files and invalid image data.
"""

from pathlib import Path
from typing import Optional, Tuple
import warnings

import numpy as np
from PIL import Image


def load_image(path: Path, mode: Optional[str] = None) -> Optional[Image.Image]:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        img = Image.open(path)
        if mode is not None and img.mode != mode:
            img = img.convert(mode)
        img.load()
        return img
    except Exception as e:
        raise ValueError(f"Failed to load image {path}: {e}")


def load_rgb_image(path: Path) -> Optional[Image.Image]:
    return load_image(path, mode="RGB")


def load_thermal_image(path: Path) -> Optional[Image.Image]:
    return load_image(path, mode="RGB")


def load_image_pair(
    rgb_path: Path,
    thermal_path: Path,
) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    rgb_img = load_rgb_image(rgb_path)
    thermal_img = load_thermal_image(thermal_path)
    return rgb_img, thermal_img


def get_image_info(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        img = Image.open(path)
        info = {
            "path": str(path),
            "width": img.size[0],
            "height": img.size[1],
            "mode": img.mode,
            "format": img.format,
            "extension": path.suffix,
            "corrupt": False,
        }
        img.close()
        return info
    except Exception:
        return {
            "path": str(path),
            "width": None,
            "height": None,
            "mode": None,
            "format": None,
            "extension": path.suffix,
            "corrupt": True,
        }
