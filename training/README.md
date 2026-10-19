# Multimodal RGB-Thermal Training Workspace

Self-contained training pipeline for LLVIP dataset. Copy this entire `training/` folder to any GPU-enabled system.

## Structure

```
training/
├── configs/
│   └── config.py          # ALL paths & hyperparameters (EDIT THIS FIRST)
├── preprocessing/
│   ├── __init__.py
│   └── convert.py         # VOC XML -> YOLO format (train/val/test splits)
├── models/
│   └── __init__.py        # Model definitions (RGB, Thermal, Fusion variants)
├── scripts/
│   └── train.py           # Training entry points
├── evaluation/
│   └── evaluate.py        # Evaluation with metrics + comparison
├── utils/
│   ├── __init__.py
│   └── export.py          # Export to ONNX/TorchScript, create deployment packages
├── main.py                # Full pipeline: preprocess -> train -> evaluate -> export
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Configure Paths
Edit `configs/config.py`:
```python
DATA_ROOT = Path("/path/to/your/LLVIP")  # Must contain visible/, infrared/, Annotations/
WORK_DIR = Path("/path/to/output/workspace")
```

Or set environment variable:
```bash
export LLVIP_DATASET_ROOT=/path/to/LLVIP
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Full Pipeline
```bash
# All modalities (RGB + Thermal + Fusion)
python -m training.main

# Single modality
python -m training.main --modality rgb
python -m training.main --modality thermal
python -m training.main --modality fusion

# Resume from existing weights (skip training)
python -m training.main --skip-training

# Run only specific step
python -m training.main --only preprocessing
python -m training.main --only evaluation --modality all
```

## What Each Step Does

| Step | Script | Output |
|------|--------|--------|
| **Preprocessing** | `preprocessing/convert.py` | `WORK_DIR/datasets/{rgb,thermal,fusion}_yolo/` with `data.yaml` |
| **Training** | `scripts/train.py` | `WORK_DIR/outputs/{rgb,thermal,fusion}/yolov8n/train/weights/best.pt` |
| **Evaluation** | `evaluation/evaluate.py` | `test_results.json/csv` + comparison table |
| **Export** | `utils/export.py` | ONNX, TorchScript + deployment zip |

## Expected Dataset Structure (LLVIP)

```
LLVIP/
├── visible/
│   ├── train/      # 12,025 RGB images
│   └── test/       # 3,463 RGB images
├── infrared/
│   ├── train/      # 12,025 thermal images
│   └── test/       # 3,463 thermal images
└── Annotations/    # 15,488 VOC XML files
```

## Key Features

- **Reproducible splits**: `seed=42`, val=10% capped at 500 (matches repo's Phase 3 audit)
- **Handles known issues**: Skips 2 empty annotations (100030, 100033) and 2 zero-width boxes (020118, 020236)
- **Three fusion modes**:
  - Early fusion: 6-channel input (RGB+thermal concatenated)
  - Late fusion: Ensemble of two models
  - Mid fusion: Cross-attention between feature pyramids
- **Full metrics**: Precision, Recall, F1, mAP@50, mAP@50:95, inference time
- **Deployment ready**: Exports to ONNX/TorchScript, creates inference script + package

## Outputs

After running, `WORK_DIR/` contains:
```
training_workspace/
├── datasets/
│   ├── rgb_yolo/          # YOLO format RGB dataset
│   ├── thermal_yolo/      # YOLO format thermal dataset
│   └── fusion_yolo/       # Paired RGB+thermal for fusion
├── outputs/
│   ├── rgb/yolov8n/train/
│   │   ├── weights/best.pt
│   │   ├── weights/last.pt
│   │   ├── results.csv
│   │   ├── test_results.json
│   │   └── exports/       # ONNX, TorchScript
│   ├── thermal/yolov8n/train/...
│   └── fusion/yolov8n/train/...
└── comparison/
    ├── comparison.csv
    └── comparison.json
```

## Colab Usage

1. Upload `training/` folder to Colab (or clone repo)
2. Mount Drive: `from google.colab import drive; drive.mount('/content/drive')`
3. Set `DATA_ROOT = Path("/content/drive/MyDrive/LLVIP")` in `configs/config.py`
4. Runtime → Change runtime type → **T4 GPU**
5. Run: `python -m training.main`

## Reproducing Paper Results

The config matches the repo's Phase 4/5/6:
- Model: YOLOv8n (COCO pretrained)
- Image size: 640×640
- Batch: 16 (reduce to 8 if OOM)
- Epochs: 50 (paper used 100, but 50 converges well)
- Seed: 42 (exact same train/val split as audit)
- Device: GPU (set `DEVICE = "0"`)

Expected mAP@50 (from 2-epoch CPU baseline):
- RGB: ~0.68 → expect 0.75-0.80 at 50 epochs
- Thermal: ~0.80 → expect 0.85-0.90 at 50 epochs
- Fusion: should exceed both

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA OOM | Reduce `BATCH_SIZE` to 8, or `IMG_SIZE` to 320 |
| Dataset not found | Check `DATA_ROOT` in config.py |
| Import errors | Run from project root: `python -m training.main` |
| Missing weights | Run without `--skip-training` first |

## License

Part of the Multimodal RGB-Thermal Human Detection project.