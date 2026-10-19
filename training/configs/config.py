"""
Configuration for portable training on any GPU system.
All paths are relative to this training/ directory.
Modify DATA_ROOT to point to your LLVIP dataset location.
"""

from pathlib import Path
import os

# ============================================================
# PATH CONFIGURATION - EDIT THESE FOR YOUR SYSTEM
# ============================================================
# Root of the LLVIP dataset (must contain visible/, infrared/, Annotations/)
DATA_ROOT = Path(os.environ.get("LLVIP_DATASET_ROOT", "/path/to/LLVIP"))

# Where to write the YOLO-format dataset and training outputs
WORK_DIR = Path(__file__).resolve().parent.parent / "training_workspace"

# Derived paths (do not edit)
VISIBLE_TRAIN  = DATA_ROOT / "visible" / "train"
VISIBLE_TEST   = DATA_ROOT / "visible" / "test"
INFRARED_TRAIN = DATA_ROOT / "infrared" / "train"
INFRARED_TEST  = DATA_ROOT / "infrared" / "test"
ANNOTATIONS    = DATA_ROOT / "Annotations"

# Output directories
YOLO_RGB_DIR    = WORK_DIR / "datasets" / "rgb_yolo"
YOLO_THERMAL_DIR = WORK_DIR / "datasets" / "thermal_yolo"
YOLO_FUSION_DIR  = WORK_DIR / "datasets" / "fusion_yolo"

OUTPUT_RGB    = WORK_DIR / "outputs" / "rgb"
OUTPUT_THERMAL = WORK_DIR / "outputs" / "thermal"
OUTPUT_FUSION  = WORK_DIR / "outputs" / "fusion"

# ============================================================
# TRAINING HYPERPARAMETERS (matching the repo's Phase 4/5/6)
# ============================================================
MODEL_NAME      = "yolov8n"       # or "yolov8s"
EPOCHS          = 50
BATCH_SIZE      = 16
IMG_SIZE        = 640
SEED            = 42
VAL_RATIO       = 0.1
MAX_VAL_IMAGES  = 500
PATIENCE        = 10
NUM_WORKERS     = 4
DEVICE          = "0"             # "0" for first GPU, "cpu" for CPU
PRETRAINED      = True

# ============================================================
# DATASET STATISTICS (from Phase 3 audit)
# ============================================================
NUM_VISIBLE_TRAIN  = 12025
NUM_VISIBLE_TEST   = 3463
NUM_INFRARED_TRAIN = 12025
NUM_INFRARED_TEST  = 3463
NUM_ANNOTATIONS    = 15488
IMG_WIDTH  = 1280
IMG_HEIGHT = 1024
NUM_CLASSES = 1
CLASS_NAMES = ["person"]

# Known issues to skip (from audit)
EMPTY_ANNOTATIONS = {"100030", "100033"}
ZERO_WIDTH_BOXES  = {"020118", "020236"}

# ============================================================
# FUSION MODEL CONFIG
# ============================================================
FUSION_METHOD = "early"   # "early" (concat), "late" (ensemble), "mid" (attention)
FUSION_CHANNELS = 6       # RGB(3) + Thermal(3) for early fusion

# ============================================================
# EXPORT SETTINGS
# ============================================================
EXPORT_FORMATS = ["onnx", "torchscript"]  # formats to export best.pt to
EXPORT_IMG_SIZE = 640