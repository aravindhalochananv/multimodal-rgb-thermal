"""
Model definitions for RGB, thermal, and fusion experiments.
Uses Ultralytics YOLOv8 as base with custom fusion heads.
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv, C2f, SPPF, Detect
from typing import Optional, List


def get_yolo_model(model_name: str = "yolov8n", num_classes: int = 1,
                   in_channels: int = 3, pretrained: bool = True) -> YOLO:
    """Get a YOLOv8 model with specified input channels."""
    model = YOLO(f"{model_name}.pt") if pretrained else YOLO(f"{model_name}.yaml")

    if in_channels != 3 and pretrained:
        # Modify first layer for different input channels
        model.model.model[0] = Conv(in_channels, model.model.model[0].conv.out_channels,
                                    model.model.model[0].conv.kernel_size,
                                    model.model.model[0].conv.stride,
                                    model.model.model[0].conv.padding)

    # Update detection head for num_classes
    if hasattr(model.model, 'model') and model.model.model[-1].nc != num_classes:
        model.model.model[-1].nc = num_classes
        # Reinitialize detection head
        m = model.model.model[-1]
        m.no = num_classes + m.reg_max * 4
        m.stride = torch.tensor([8, 16, 32])
        m.anchors = torch.zeros(m.nl, m.na, 2)

    return model


class EarlyFusionYOLO(nn.Module):
    """
    Early fusion: concatenate RGB (3ch) + Thermal (3ch) = 6 channels input.
    Uses a single YOLOv8 backbone with modified first layer.
    """
    def __init__(self, model_name: str = "yolov8n", num_classes: int = 1,
                 pretrained: bool = True):
        super().__init__()
        self.model = get_yolo_model(model_name, num_classes, in_channels=6, pretrained=pretrained)

    def forward(self, rgb, thermal):
        # rgb: [B, 3, H, W], thermal: [B, 3, H, W] (or [B, 1, H, W])
        if thermal.shape[1] == 1:
            thermal = thermal.repeat(1, 3, 1, 1)  # expand to 3 channels
        x = torch.cat([rgb, thermal], dim=1)  # [B, 6, H, W]
        return self.model(x)

    def to(self, device):
        self.model.to(device)
        return super().to(device)

    def train(self, mode=True):
        self.model.train(mode)
        return super().train(mode)

    def eval(self):
        self.model.eval()
        return super().eval()

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict)


class LateFusionEnsemble(nn.Module):
    """
    Late fusion: two separate YOLOv8 models (RGB + Thermal), ensemble predictions.
    """
    def __init__(self, model_name: str = "yolov8n", num_classes: int = 1,
                 pretrained: bool = True):
        super().__init__()
        self.rgb_model = get_yolo_model(model_name, num_classes, in_channels=3, pretrained=pretrained)
        self.thermal_model = get_yolo_model(model_name, num_classes, in_channels=3, pretrained=pretrained)

    def forward(self, rgb, thermal):
        rgb_out = self.rgb_model(rgb)
        thermal_out = self.thermal_model(thermal)
        return rgb_out, thermal_out

    def ensemble_predict(self, rgb, thermal, conf_thres=0.25, iou_thres=0.5):
        """Ensemble predictions with NMS."""
        from ultralytics.utils.ops import non_max_suppression
        rgb_out = self.rgb_model(rgb)
        thermal_out = self.thermal_model(thermal)

        # Apply NMS to each
        rgb_preds = non_max_suppression(rgb_out[0], conf_thres, iou_thres)
        thermal_preds = non_max_suppression(thermal_out[0], conf_thres, iou_thres)

        # Simple box averaging for overlapping detections
        fused = []
        for rp, tp in zip(rgb_preds, thermal_preds):
            if rp is None and tp is None:
                fused.append(None)
            elif rp is None:
                fused.append(tp)
            elif tp is None:
                fused.append(rp)
            else:
                # Concatenate and NMS again
                combined = torch.cat([rp, tp], dim=0)
                fused.append(non_max_suppression(combined.unsqueeze(0), conf_thres, iou_thres)[0])
        return fused


class AttentionFusion(nn.Module):
    """
    Mid-level fusion: cross-attention between RGB and thermal features.
    """
    def __init__(self, model_name: str = "yolov8n", num_classes: int = 1,
                 pretrained: bool = True):
        super().__init__()
        # Use two backbones, fuse at neck level
        self.rgb_backbone = get_yolo_model(model_name, num_classes, in_channels=3, pretrained=pretrained)
        self.thermal_backbone = get_yolo_model(model_name, num_classes, in_channels=3, pretrained=pretrained)

        # Cross-attention modules for each scale
        self.attention_modules = nn.ModuleList([
            CrossAttention(256),  # P3
            CrossAttention(512),  # P4
            CrossAttention(1024), # P5
        ])

    def forward(self, rgb, thermal):
        # Extract multi-scale features from both backbones
        rgb_feats = self._extract_features(self.rgb_backbone, rgb)
        thermal_feats = self._extract_features(self.thermal_backbone, thermal)

        # Fuse at each scale
        fused_feats = []
        for i, (rf, tf) in enumerate(zip(rgb_feats, thermal_feats)):
            fused = self.attention_modules[i](rf, tf)
            fused_feats.append(fused)

        # Pass through detection head
        return self.rgb_backbone.model.model[-1](fused_feats)

    def _extract_features(self, model, x):
        """Extract P3, P4, P5 features from YOLO backbone."""
        feats = []
        for i, m in enumerate(model.model.model[:-1]):  # exclude detect head
            x = m(x)
            if i in [4, 6, 9]:  # P3, P4, P5 indices for yolov8n
                feats.append(x)
        return feats


class CrossAttention(nn.Module):
    """Cross-attention between two feature maps."""
    def __init__(self, channels: int):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x1, x2):
        # x1: RGB features, x2: thermal features
        B, C, H, W = x1.shape
        q = self.q(x1).view(B, C, -1).transpose(1, 2)  # [B, HW, C]
        k = self.k(x2).view(B, C, -1)                   # [B, C, HW]
        v = self.v(x2).view(B, C, -1).transpose(1, 2)   # [B, HW, C]

        attn = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).view(B, C, H, W)
        return self.out(out) + x1  # residual


def create_model(modality: str, model_name: str = "yolov8n",
                 num_classes: int = 1, pretrained: bool = True):
    """Factory function to create model by modality."""
    if modality == "rgb":
        return get_yolo_model(model_name, num_classes, in_channels=3, pretrained=pretrained)
    elif modality == "thermal":
        return get_yolo_model(model_name, num_classes, in_channels=3, pretrained=pretrained)
    elif modality == "fusion_early":
        return EarlyFusionYOLO(model_name, num_classes, pretrained)
    elif modality == "fusion_late":
        return LateFusionEnsemble(model_name, num_classes, pretrained)
    elif modality == "fusion_mid":
        return AttentionFusion(model_name, num_classes, pretrained)
    else:
        raise ValueError(f"Unknown modality: {modality}")


if __name__ == "__main__":
    # Test model creation
    m = create_model("rgb", "yolov8n", 1)
    print(f"RGB model: {type(m)}")

    m = create_model("thermal", "yolov8n", 1)
    print(f"Thermal model: {type(m)}")

    m = create_model("fusion_early", "yolov8n", 1)
    print(f"Early fusion model: {type(m)}")

    m = create_model("fusion_late", "yolov8n", 1)
    print(f"Late fusion model: {type(m)}")

    m = create_model("fusion_mid", "yolov8n", 1)
    print(f"Mid fusion model: {type(m)}")