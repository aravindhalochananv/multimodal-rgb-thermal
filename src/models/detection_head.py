"""
Shared detection head and post-processing components for YOLOv8 models.
Provides bounding box decoding, NMS filtering, and class mapping utilities.
"""

from typing import List, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from ultralytics.nn.modules import Detect


class MultimodalDetectionHead(nn.Module):
    """
    Detection head wrapper for multimodal predictions.
    Adapts standard YOLOv8 detection head outputs for single-class person detection.
    """
    def __init__(self, num_classes: int = 1, reg_max: int = 16):
        super().__init__()
        self.nc = num_classes
        self.reg_max = reg_max
        self.no = num_classes + reg_max * 4

    @staticmethod
    def filter_boxes(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies confidence thresholding and Non-Maximum Suppression (NMS).
        
        Args:
            boxes: [N, 4] bounding boxes in (x1, y1, x2, y2) format
            scores: [N] confidence scores
            conf_threshold: Minimum confidence score
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Tuple of filtered (boxes, scores)
        """
        keep_mask = scores >= conf_threshold
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]

        if boxes.shape[0] == 0:
            return boxes, scores

        from torchvision.ops import nms
        keep_idx = nms(boxes, scores, iou_threshold)
        return boxes[keep_idx], scores[keep_idx]
