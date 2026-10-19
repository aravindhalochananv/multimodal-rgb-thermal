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

### 7. Train Models (Phase 4+)

#### Single-Modality Training

```bash
# RGB-only baseline
python -m src.training.train_rgb --epochs 100 --batch-size 8 --img-size 640

# Thermal-only baseline
python -m src.training.train_thermal --epochs 100 --batch-size 8 --img-size 640
```

#### Generic Training (more control)

```bash
python -m src.training.train_yolo \
  --modality rgb \
  --model yolov8n \
  --epochs 100 \
  --batch-size 8 \
  --img-size 640
```

Options: `--modality` (rgb/thermal), `--model` (yolov8n/yolov8s), `--epochs`, `--batch-size`, `--img-size`, `--patience`, `--force-rebuild`.

### 8. Evaluate Models (Phase 7)

```bash
# Evaluate the RGB baseline model
python -m src.evaluation.compare
```

Individual evaluation:
```python
from src.evaluation.evaluate_yolo import evaluate
results = evaluate(modality="rgb", model_name="yolov8n")
```

Results are saved per experiment as `outputs/{rgb,thermal}/{model_name}/test_results.json` and `test_results.csv`.

### 9. Compare Experiments (Phase 8)

```bash
python -m src.evaluation.compare
```

Outputs:
- `outputs/comparison/comparison.csv` — Metrics table
- `outputs/comparison/comparison.json` — Raw results
- `outputs/comparison/comparison.txt` — Summary report

### 10. Interactive React Web Studio & Dashboard

Launch the full-stack web application to test models, run 3-way comparisons, and inspect benchmark analytics in an interactive UI:

```bash
# 1-Click Launch (starts FastAPI server and opens React UI in your browser)
python run_app.py

# Or on Windows, double-click:
start_app.bat
```

* **Web Studio URL:** `http://127.0.0.1:8000`
* **API Documentation:** `http://127.0.0.1:8000/docs`

#### Key Web Studio Capabilities
1. **Colab Model Selector:** Switch between RGB-Only, Thermal-Only, and 50:50 Multimodal Fusion YOLOv8n models with real-time parameter badge displays.
2. **Interactive Fusion Ratio Slider ($\alpha$):** Dynamically blend visible and infrared streams from 0% RGB / 100% Thermal up to 100% RGB / 0% Thermal.
3. **Triple Comparison View:** Run all 3 Colab models simultaneously on paired inputs with side-by-side detection panels, confidence scores, and latency statistics.
4. **Pre-Loaded Test Scenes & Custom Uploads:** Select from 20 LLVIP test pairs or drag-and-drop custom images.
5. **Benchmark Analytics:** Interactive table of Precision, Recall, F1, mAP@50, mAP@50:95, and embedded high-resolution training curves.
6. **Qualitative Gallery:** Walkthrough of challenging nighttime scenes (pedestrians in dark shadows, headlight glare, distance).

---

## Experimental Results (10-Epoch GPU Full Evaluation)

All three models were trained for 10 epochs with GPU acceleration on Google Colab and evaluated on the official LLVIP test set (3,463 unseen paired images).

### 3-Way Benchmark Comparison Table

| Modality / Experiment | Precision | Recall | F1-Score | mAP@50 | mAP@50:95 | Inference Time | FPS (CPU) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **RGB-Only Baseline** | 0.8904 | 0.7870 | 0.8356 | 0.8596 | 0.4692 | 54.07 ms | 18.5 |
| **Thermal-Only Baseline** | **0.9267** | **0.8418** | **0.8822** | **0.9222** | **0.5658** | 52.15 ms | 19.2 |
| **50:50 Multimodal Fusion** | 0.9218 | 0.8295 | 0.8732 | 0.9150 | 0.5505 | **38.19 ms** | **26.2** |

### Key Experimental Insights
* **Thermal Superiority:** Thermal-only outperforms RGB-only by **+6.26% mAP@50** and **+9.66% mAP@50:95**, demonstrating that infrared body heat radiation penetrates nighttime shadows and headlight glare far more effectively than visible light.
* **Multimodal Advantage:** 50:50 Multimodal Fusion outperforms RGB-only by **+5.54% mAP@50** and **+8.13% mAP@50:95**, significantly reducing false negatives while maintaining high FPS (26.2 FPS on CPU).
* **Fusion Dynamics:** Naive 50:50 pixel fusion slightly attenuates thermal peak contrast (-0.72% vs. pure thermal) due to blending with dark RGB noise. This confirms that while pixel fusion greatly aids RGB visibility, advanced feature-level cross-attention is recommended for ultimate accuracy.

---

## RGB-Only Baseline (Phase 4)

* **Architecture:** YOLOv8n (3.0M parameters)
* **Training:** 10 epochs on NVIDIA GPU, batch size 16, image size 320x320
* **Test Metrics:** Precision: 0.8904 | Recall: 0.7870 | F1: 0.8356 | mAP@50: 0.8596 | mAP@50:95: 0.4692
* **Artifacts:** `outputs/rgb/yolov8n/train/weights/best.pt`, `outputs/rgb/predictions/`

## Thermal-Only Baseline (Phase 5)

* **Architecture:** YOLOv8n (3.0M parameters)
* **Training:** 10 epochs on NVIDIA GPU, batch size 16, image size 320x320
* **Test Metrics:** Precision: 0.9267 | Recall: 0.8418 | F1: 0.8822 | mAP@50: 0.9222 | mAP@50:95: 0.5658
* **Artifacts:** `outputs/thermal/yolov8n/train/weights/best.pt`, `outputs/thermal/predictions/`

## Multimodal RGB-Thermal Fusion (Phase 6)

* **Architecture:** YOLOv8n with 50:50 weighted pixel fusion ($I_{\text{fused}} = 0.5 \cdot I_{\text{rgb}} + 0.5 \cdot I_{\text{thermal}}$)
* **Extended Architectures:** `src/models/fusion_model.py` provides `LateFusionEnsemble` and `EarlyFusionYOLO`
* **Training:** 10 epochs on NVIDIA GPU, batch size 16, image size 320x320
* **Test Metrics:** Precision: 0.9218 | Recall: 0.8295 | F1: 0.8732 | mAP@50: 0.9150 | mAP@50:95: 0.5505
* **Artifacts:** `outputs/multimodal/yolov8n/train/weights/best.pt`, `outputs/multimodal/predictions/`

## Comparison & Visualizations (Phase 8)

* `outputs/comparison/comparison.csv`: Summary metrics table
* `outputs/comparison/metrics_comparison_bar.png`: Multi-metric bar plot
* `outputs/comparison/map_comparison.png`: mAP@50 and mAP@50:95 comparison plot
* `outputs/comparison/training_curves_comparison.png`: 10-epoch validation curves across all modalities

## Inference & Qualitative Detections (Phase 9)

Run inference or generate 3-way side-by-side comparative detection panels:

```bash
# 3-Way Comparative Detection Panel (RGB vs Thermal vs Fusion)
python -m src.inference.predict --compare --rgb data/LLVIP/visible/test/190001.jpg --thermal data/LLVIP/infrared/test/190001.jpg

# Multimodal Fusion Prediction
python -m src.inference.predict --modality multimodal --rgb data/LLVIP/visible/test/190001.jpg --thermal data/LLVIP/infrared/test/190001.jpg
```

Comparative panels are saved to `outputs/comparison/predictions/`.

---

## Dataset Audit Results (Phase 3)

The dataset audit was verified via `src/data/audit.py` and outputs are in `outputs/dataset_audit/`:
* **Total Checks:** 35 — 31 PASS, 1 FAIL (zero-width bboxes skipped in conversion), 1 WARN, 2 INFO.
* **Pairs:** 15,488 verified pairs (100% matched, zero leakage between 12,025 train and 3,463 test).

## Phases

1. **Phase 1** — Environment and project structure ✅
2. **Phase 2** — LLVIP dataset setup and organization ✅
3. **Phase 3** — Dataset audit and RGB/thermal pairing verification ✅
4. **Phase 4** — RGB-only baseline training & evaluation ✅
5. **Phase 5** — Thermal-only baseline training & evaluation ✅
6. **Phase 6** — RGB + Thermal multimodal fusion model ✅
7. **Phase 7** — Comprehensive test set evaluation ✅
8. **Phase 8** — Comparative analysis and visualization ✅
9. **Phase 9** — Final inference CLI and qualitative demo panels ✅

## Requirements

See `requirements.txt`. Python 3.11+, PyTorch (CPU).

## Evaluation Metrics

- Precision, Recall, F1-score
- mAP@50, mAP@50:95
- Inference time

## Author

University Machine Learning Project
