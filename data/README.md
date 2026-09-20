# Data Directory

This directory contains project data subdirectories but **does NOT include the LLVIP dataset**.

## LLVIP Dataset

The LLVIP (Visible-Infrared Paired Dataset for Low-Light Vision) dataset is **not included** in this GitHub repository. It is a separate, external dataset that must be obtained independently.

### Configure Your Local LLVIP Path

Before running any pipeline, set the `LLVIP_DATASET_ROOT` environment variable to your local LLVIP dataset root directory:

```bash
# Windows (PowerShell)
$env:LLVIP_DATASET_ROOT = "C:/path/to/LLVIP"

# Windows (CMD)
set LLVIP_DATASET_ROOT=C:/path/to/LLVIP

# Linux/macOS
export LLVIP_DATASET_ROOT=/path/to/LLVIP
```

Or edit the default in `src/config.py`.

### Expected Dataset Structure

```
LLVIP/
├── visible/
│   ├── train/    # RGB training images
│   └── test/     # RGB test images
├── infrared/
│   ├── train/    # Thermal training images
│   └── test/     # Thermal test images
└── Annotations/  # PASCAL VOC XML annotations
```

## Project Directories

| Directory | Purpose |
|---|---|
| `raw/` | Intentionally empty — LLVIP remains external |
| `processed/` | For processed train/val/test data (generated during training) |
| `dataset_audit/` | Phase 3 audit reports (`dataset_audit.json`, `dataset_audit.csv`, `pairing_report.csv`) |

## Pairing Convention

RGB, thermal, and annotation files are paired by filename stem:

```
visible/{N}.jpg      ← RGB image
infrared/{N}.jpg      ← Thermal image
Annotations/{N}.xml   ← Annotation
```
