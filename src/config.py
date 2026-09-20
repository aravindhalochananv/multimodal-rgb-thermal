"""
Project configuration for Multimodal RGB-Thermal Human Detection.

All paths, hyperparameters, and experiment settings are centralized here.
Modify this file to adjust project-wide settings.

Dataset path configuration:
The LLVIP dataset path MUST be provided via the LLVIP_DATASET_ROOT
environment variable. Set it before running any pipeline:

    import os
    os.environ["LLVIP_DATASET_ROOT"] = "C:/path/to/LLVIP"

Or set it in your shell before running Python:
    Windows (PowerShell):  $env:LLVIP_DATASET_ROOT = "C:/path/to/LLVIP"
    Windows (CMD):         set LLVIP_DATASET_ROOT=C:/path/to/LLVIP
    Linux/macOS:           export LLVIP_DATASET_ROOT=/path/to/LLVIP

If LLVIP_DATASET_ROOT is not set, a configuration error will be raised.
See .env.example for the expected format.
"""

import os
from pathlib import Path


class ConfigurationError(Exception):
    pass


# ---------------------------------------------------------------------------
# Project root (auto-resolved relative to this file)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# LLVIP Dataset path
# MUST be provided via LLVIP_DATASET_ROOT environment variable.
# ---------------------------------------------------------------------------
_llvip_path = os.environ.get("LLVIP_DATASET_ROOT")
if not _llvip_path:
    raise ConfigurationError(
        "\n\n"
        "╔══════════════════════════════════════════════════════════════════╗\n"
        "║  CONFIGURATION ERROR: LLVIP_DATASET_ROOT is not set            ║\n"
        "║                                                                  ║\n"
        "║  The LLVIP dataset path must be configured before running any    ║\n"
        "║  pipeline. Set the LLVIP_DATASET_ROOT environment variable to    ║\n"
        "║  the root directory of your locally downloaded LLVIP dataset.    ║\n"
        "║                                                                  ║\n"
        "║  Examples:                                                       ║\n"
        "║    Windows (PowerShell): $env:LLVIP_DATASET_ROOT = \"C:/LLVIP\"  ║\n"
        "║    Windows (CMD):        set LLVIP_DATASET_ROOT=C:/LLVIP        ║\n"
        "║    Linux/macOS:          export LLVIP_DATASET_ROOT=/path/LLVIP  ║\n"
        "║                                                                  ║\n"
        "║  See .env.example for the expected format.                       ║\n"
        "╚══════════════════════════════════════════════════════════════════╝\n\n"
    )
DATASET_ROOT = Path(_llvip_path)

# RGB / Visible images
DATA_VISIBLE_ROOT = DATASET_ROOT / "visible"
DATA_VISIBLE_TRAIN = DATA_VISIBLE_ROOT / "train"
DATA_VISIBLE_TEST = DATA_VISIBLE_ROOT / "test"

# Thermal / Infrared images
DATA_INFRARED_ROOT = DATASET_ROOT / "infrared"
DATA_INFRARED_TRAIN = DATA_INFRARED_ROOT / "train"
DATA_INFRARED_TEST = DATA_INFRARED_ROOT / "test"

# Annotations (PASCAL VOC XML)
DATA_ANNOTATIONS = DATASET_ROOT / "Annotations"

# ---------------------------------------------------------------------------
# Internal project directories (for processed data, audits, outputs)
# ---------------------------------------------------------------------------
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_TRAIN = DATA_PROCESSED / "train"
DATA_VAL = DATA_PROCESSED / "val"
DATA_TEST = DATA_PROCESSED / "test"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUTPUTS = PROJECT_ROOT / "outputs"
OUTPUT_RGB = OUTPUTS / "rgb"
OUTPUT_THERMAL = OUTPUTS / "thermal"
OUTPUT_MULTIMODAL = OUTPUTS / "multimodal"
OUTPUT_COMPARISON = OUTPUTS / "comparison"

DATA_AUDIT = OUTPUTS / "dataset_audit"
LOG_DIR = OUTPUTS / "logs"

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
MODEL_NAME = "yolov8n"
NUM_CLASSES = 1
TARGET_CLASS = "person"

# ---------------------------------------------------------------------------
# Image settings (from LLVIP inspection)
# ---------------------------------------------------------------------------
IMG_SIZE = 640
IMG_WIDTH = 1280
IMG_HEIGHT = 1024
IMG_CHANNELS_RGB = 3
IMG_CHANNELS_THERMAL = 1
IMG_FORMAT = ".jpg"
ANN_FORMAT = ".xml"

# ---------------------------------------------------------------------------
# LLVIP dataset statistics (from Phase 3 audit — actual counts)
# ---------------------------------------------------------------------------
NUM_VISIBLE_TRAIN = 12025
NUM_VISIBLE_TEST = 3463
NUM_INFRARED_TRAIN = 12025
NUM_INFRARED_TEST = 3463
NUM_TOTAL_PAIRS = 15488
NUM_ANNOTATIONS = 15488

# ---------------------------------------------------------------------------
# LLVIP pairing convention (from Phase 3 audit)
# RGB, thermal, and annotation files share the same stem (filename without
# extension). Example: visible/010001.jpg, infrared/010001.jpg,
# Annotations/010001.xml. Pairing key = stem.
# ---------------------------------------------------------------------------
PAIRING_KEY = "stem"

# ---------------------------------------------------------------------------
# Train/test split (LLVIP provides predefined split — do not re-split)
# ---------------------------------------------------------------------------
SPLIT = "predefined"
TRAIN_DIRS = [DATA_VISIBLE_TRAIN, DATA_INFRARED_TRAIN]
TEST_DIRS = [DATA_VISIBLE_TEST, DATA_INFRARED_TEST]
VAL_RATIO = 0.0

# ---------------------------------------------------------------------------
# Training defaults (populated per experiment during Phases 4–6)
# ---------------------------------------------------------------------------
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.001
SEED = 42

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = "cuda" if False else "cpu"


class ConfigurationError(Exception):
    pass
