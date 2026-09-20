# Multimodal RGB-Thermal Human Detection Under Challenging Visibility Conditions

## Project Overview

A multimodal machine-learning system for human/pedestrian detection using paired RGB/visible and thermal/infrared images from the LLVIP (Visible-Infrared Paired Dataset for Low-Light Vision) dataset.

This project investigates whether combining RGB and thermal information improves human detection under challenging visibility conditions.

## Experiments

| Experiment | Input | Purpose |
|---|---|---|
| RGB-Only | RGB/visible image | Establish RGB baseline |
| Thermal-Only | Thermal/infrared image | Establish thermal baseline |
| RGB + Thermal | Paired RGB + thermal | Investigate multimodal fusion benefit |

## Dataset

**LLVIP** — Visible-Infrared Paired Dataset for Low-Light Vision.

> **The dataset is NOT included in this GitHub repository.**
> You must obtain the LLVIP dataset separately and configure the local path before running any pipeline.

### Expected Dataset Structure

```
LLVIP/
├── visible/
│   ├── train/          # 12,025 RGB training images
│   └── test/           # 3,463 RGB test images
├── infrared/
│   ├── train/          # 12,025 thermal training images
│   └── test/           # 3,463 thermal test images
└── Annotations/         # 15,488 PASCAL VOC XML annotation files
```

### Pairing Convention

RGB, thermal, and annotation files are paired by **filename stem**:

```
visible/{N}.jpg      ← RGB image
infrared/{N}.jpg      ← Thermal image
Annotations/{N}.xml   ← Annotation
```

Example: `010001.jpg` (RGB) + `010001.jpg` (thermal) + `010001.xml` (annotation)

### Verified Dataset Statistics (Phase 3 Audit)

| Property | Value |
|---|---|
| RGB images | 15,488 |
| Thermal images | 15,488 |
| Annotations | 15,488 |
| Valid RGB-thermal pairs | 15,488 (100%) |
| Person objects (total) | 42,437 |
| Image dimensions | 1280×1024 |
| Image format | JPEG |
| Annotation format | PASCAL VOC XML |
| Train/test split | Predefined (12,025 train / 3,463 test) |
| RGB/thermal pairing mismatches | 0 |
| Train/test filename overlap | 0 (no leakage) |

### Known Data-Quality Issues (Identified, Not Yet Fixed)

| Issue | Files | Description |
|---|---|---|
| Empty annotations | `100030.xml`, `100033.xml` | Annotation file contains no `<object>` elements |
| Zero-width bounding boxes | `020118.xml`, `020236.xml` | `xmin >= xmax` in bounding box |

These issues will be handled during preprocessing (Phases 4–6). They have **not** been fixed.

## Project Structure

```
multimodal-rgb-thermal/
│
├── data/
│   ├── raw/                  # Intentionally empty — LLVIP is external
│   │   ├── rgb/
│   │   ├── thermal/
│   │   └── annotations/
│   ├── processed/            # For processed data (post-training)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── dataset_audit/        # Audit reports (Phase 3 outputs)
│   └── README.md             # Data directory documentation
│
├── src/
│   ├── config.py             # Centralized configuration
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py # Image loading utilities
│   │   ├── pairing.py        # RGB/thermal pairing logic
│   │   ├── preprocessing.py  # Preprocessing pipeline
│   │   └── audit.py          # Dataset audit pipeline
│   ├── models/               # Model definitions (Phase 4+)
│   ├── training/             # Training scripts (Phase 4+)
│   ├── evaluation/           # Evaluation scripts (Phase 7+)
│   ├── inference/            # Inference script (Phase 9+)
│   └── utils/                # Utilities
│
├── notebooks/
│   └── dataset_audit.ipynb   # Interactive audit notebook
│
├── outputs/
│   ├── dataset_audit/        # Audit results (Phase 3)
│   ├── rgb/                  # RGB model outputs (Phase 4+)
│   ├── thermal/              # Thermal model outputs (Phase 5+)
│   ├── multimodal/           # Multimodal outputs (Phase 6+)
│   └── comparison/           # Comparison visualizations (Phase 8+)
│
├── .env.example               # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd multimodal-rgb-thermal
```

### 2. Create a Python 3.11 Virtual Environment

```bash
python -m venv .venv
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.\.venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Set LLVIP Dataset Path

**Option A: Environment variable (recommended)**

Set `LLVIP_DATASET_ROOT` to your local LLVIP dataset root before running any script:

**Windows (PowerShell):**
```powershell
$env:LLVIP_DATASET_ROOT = "C:/path/to/LLVIP"
```

**Windows (CMD):**
```cmd
set LLVIP_DATASET_ROOT=C:/path/to/LLVIP
```

**Linux/macOS:**
```bash
export LLVIP_DATASET_ROOT=/path/to/LLVIP
```

**Option B: Edit config.py**

In `src/config.py`, modify the `DATASET_ROOT` default:

```python
DATASET_ROOT = Path(r"C:/path/to/LLVIP")
```

**Option C: Use .env file**

Copy `.env.example` to `.env` and edit the path:

```bash
cp .env.example .env
```

Then edit `.env` with your LLVIP path. A startup helper in your scripts can read it via `os.environ.get("LLVIP_DATASET_ROOT")`.

> **Important**: Never commit `.env` to version control. It is excluded by `.gitignore`.

### 5. Verify Dataset Path

```bash
python -c "import sys; sys.path.insert(0, 'src'); from config import DATASET_ROOT; print(DATASET_ROOT)"
```

### 6. Run Dataset Audit

```bash
python run_audit.py
```

Or open `notebooks/dataset_audit.ipynb` in Jupyter Lab/Notebook.

## Dataset Audit Results (Phase 3)

The dataset audit has been completed. Results are in `outputs/dataset_audit/`:

| File | Description |
|---|---|
| `dataset_audit.json` | Full structured audit results |
| `dataset_audit.csv` | Check-by-check results |
| `dataset_audit.txt` | Human-readable report |
| `pairing_report.csv` | Pairing verification summary |

**Summary**: 35 checks — 31 PASS, 1 FAIL (2 zero-width bounding boxes), 1 WARN (2 empty annotations), 2 INFO.

## Phases

1. **Phase 1** — Environment and project structure ✅
2. **Phase 2** — LLVIP dataset setup and organization ✅
3. **Phase 3** — Dataset audit and RGB/thermal pairing verification ✅
4. **Phase 4** — RGB-only baseline
5. **Phase 5** — Thermal-only baseline
6. **Phase 6** — RGB + thermal multimodal fusion
7. **Phase 7** — Evaluation
8. **Phase 8** — Comparative analysis and visualization
9. **Phase 9** — Final inference/demo

## Requirements

See `requirements.txt`. Python 3.11+, PyTorch (CPU).

## Evaluation Metrics

- Precision, Recall, F1-score
- mAP@50, mAP@50:95
- Inference time

## Author

University Machine Learning Project
