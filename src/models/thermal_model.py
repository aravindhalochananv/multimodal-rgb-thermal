"""
Thermal-only detection model based on YOLOv8.
Handles single-modality infrared/thermal spectrum human detection.
"""

from pathlib import Path
from typing import Optional, Union, List, Dict
import numpy as np
from ultralytics import YOLO

from src.config import OUTPUT_THERMAL, MODEL_NAME


class ThermalYOLOModel:
    """Wrapper for Thermal YOLOv8 pedestrian detector."""

    def __init__(self, weights_path: Optional[Union[str, Path]] = None, model_name: str = MODEL_NAME):
        self.model_name = model_name
        if weights_path is None:
            default_best = OUTPUT_THERMAL / model_name / "train" / "weights" / "best.pt"
            if default_best.exists():
                weights_path = default_best
            else:
                weights_path = f"{model_name}.pt"
        self.weights_path = Path(weights_path)
        self.model = YOLO(str(self.weights_path))

    def predict(
        self,
        source: Union[str, Path, np.ndarray],
        imgsz: int = 320,
        conf: float = 0.25,
        iou: float = 0.5,
        verbose: bool = False,
        **kwargs,
    ):
        """Run prediction on a thermal image."""
        return self.model.predict(
            source=source,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=verbose,
            **kwargs,
        )

    def val(
        self,
        data_yaml: Union[str, Path],
        split: str = "test",
        imgsz: int = 320,
        batch: int = 16,
        device: str = "cpu",
        **kwargs,
    ):
        """Run validation on test/val set."""
        return self.model.val(
            data=str(data_yaml),
            split=split,
            imgsz=imgsz,
            batch=batch,
            device=device,
            **kwargs,
        )


def get_thermal_model(weights_path: Optional[Union[str, Path]] = None) -> ThermalYOLOModel:
    """Factory function for Thermal detection model."""
    return ThermalYOLOModel(weights_path=weights_path)
