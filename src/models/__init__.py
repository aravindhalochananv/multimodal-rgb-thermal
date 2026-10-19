"""
Model package exports for RGB, Thermal, and Multimodal Detection.
"""

from src.models.detection_head import MultimodalDetectionHead
from src.models.rgb_model import RGBYOLOModel, get_rgb_model
from src.models.thermal_model import ThermalYOLOModel, get_thermal_model
from src.models.fusion_model import PixelFusionYOLO, LateFusionEnsemble, get_fusion_model

__all__ = [
    "MultimodalDetectionHead",
    "RGBYOLOModel",
    "get_rgb_model",
    "ThermalYOLOModel",
    "get_thermal_model",
    "PixelFusionYOLO",
    "LateFusionEnsemble",
    "get_fusion_model",
]
