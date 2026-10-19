"""
RGB + Thermal multimodal fusion models.
Implements:
1. PixelFusionYOLO: 50:50 weighted pixel fusion with dedicated YOLOv8n detector.
2. LateFusionEnsemble: Dual-stream late fusion with bounding box NMS ensemble.
3. EarlyFusionYOLO: 6-channel early concatenation fusion.
"""

from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

from src.config import OUTPUT_MULTIMODAL, OUTPUT_RGB, OUTPUT_THERMAL, MODEL_NAME


class PixelFusionYOLO:
    """
    Weighted pixel-level RGB-Thermal fusion model.
    Blends visible and infrared images: Fused = alpha * RGB + (1 - alpha) * Thermal,
    and runs detection using the trained multimodal YOLOv8n network.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        alpha: float = 0.5,
        model_name: str = MODEL_NAME,
    ):
        self.alpha = alpha
        self.model_name = model_name
        if weights_path is None:
            default_best = OUTPUT_MULTIMODAL / model_name / "train" / "weights" / "best.pt"
            if default_best.exists():
                weights_path = default_best
            else:
                weights_path = f"{model_name}.pt"
        self.weights_path = Path(weights_path)
        self.model = YOLO(str(self.weights_path))

    def blend(
        self,
        rgb_input: Union[str, Path, np.ndarray],
        thermal_input: Union[str, Path, np.ndarray],
    ) -> np.ndarray:
        """Blend RGB and Thermal images pixel-wise."""
        if isinstance(rgb_input, (str, Path)):
            rgb_img = cv2.imread(str(rgb_input))
        else:
            rgb_img = rgb_input.copy()

        if isinstance(thermal_input, (str, Path)):
            th_img = cv2.imread(str(thermal_input))
        else:
            th_img = thermal_input.copy()

        if rgb_img.shape[:2] != th_img.shape[:2]:
            th_img = cv2.resize(th_img, (rgb_img.shape[1], rgb_img.shape[0]))

        fused = cv2.addWeighted(rgb_img, self.alpha, th_img, 1.0 - self.alpha, 0)
        return fused

    def predict(
        self,
        rgb_input: Union[str, Path, np.ndarray],
        thermal_input: Union[str, Path, np.ndarray],
        imgsz: int = 320,
        conf: float = 0.25,
        iou: float = 0.5,
        verbose: bool = False,
        **kwargs,
    ):
        """Blend paired inputs and predict detections."""
        fused = self.blend(rgb_input, thermal_input)
        return self.model.predict(
            source=fused,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=verbose,
            **kwargs,
        )


class LateFusionEnsemble:
    """
    Late fusion: runs individual RGB and Thermal models, then ensembles their box predictions.
    """

    def __init__(
        self,
        rgb_weights: Optional[Union[str, Path]] = None,
        thermal_weights: Optional[Union[str, Path]] = None,
        model_name: str = MODEL_NAME,
    ):
        rgb_path = rgb_weights or (OUTPUT_RGB / model_name / "train" / "weights" / "best.pt")
        th_path = thermal_weights or (OUTPUT_THERMAL / model_name / "train" / "weights" / "best.pt")
        self.rgb_model = YOLO(str(rgb_path))
        self.thermal_model = YOLO(str(th_path))

    def predict(
        self,
        rgb_input: Union[str, Path, np.ndarray],
        thermal_input: Union[str, Path, np.ndarray],
        imgsz: int = 320,
        conf: float = 0.25,
        iou: float = 0.5,
    ) -> List[Dict]:
        """Predict with both models and merge boxes via NMS."""
        rgb_res = self.rgb_model.predict(rgb_input, imgsz=imgsz, conf=conf, verbose=False)[0]
        th_res = self.thermal_model.predict(thermal_input, imgsz=imgsz, conf=conf, verbose=False)[0]

        all_boxes = []
        all_scores = []

        if len(rgb_res.boxes) > 0:
            all_boxes.append(rgb_res.boxes.xyxy)
            all_scores.append(rgb_res.boxes.conf)

        if len(th_res.boxes) > 0:
            all_boxes.append(th_res.boxes.xyxy)
            all_scores.append(th_res.boxes.conf)

        if not all_boxes:
            return []

        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)

        from torchvision.ops import nms
        keep = nms(boxes, scores, iou)

        merged = []
        for idx in keep:
            merged.append({
                "bbox": boxes[idx].cpu().numpy().tolist(),
                "confidence": float(scores[idx].cpu().item()),
                "class": "person",
            })
        return merged


def get_fusion_model(weights_path: Optional[Union[str, Path]] = None, alpha: float = 0.5) -> PixelFusionYOLO:
    """Factory function for multimodal pixel fusion model."""
    return PixelFusionYOLO(weights_path=weights_path, alpha=alpha)
