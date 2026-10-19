"""
Main entry point for the training pipeline.
Runs preprocessing -> training -> evaluation -> export in sequence.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from training.configs.config import *
from training.preprocessing import convert_all_modalities
from training.scripts.train import train_all
from training.evaluation.evaluate import evaluate_all
from training.utils.export import export_model, create_deployment_package, download_from_colab


def run_preprocessing():
    """Step 1: Convert dataset to YOLO format."""
    print("\n" + "="*60)
    print("STEP 1: PREPROCESSING - VOC to YOLO Conversion")
    print("="*60)

    # Verify dataset exists
    for d in [VISIBLE_TRAIN, VISIBLE_TEST, INFRARED_TRAIN, INFRARED_TEST, ANNOTATIONS]:
        if not d.exists():
            raise FileNotFoundError(f"Missing dataset directory: {d}")

    print(f"Dataset root: {DATA_ROOT}")
    print(f"Workspace: {WORK_DIR}")

    results = convert_all_modalities(sys.modules['training.configs.config'])

    print("\nPreprocessing complete!")
    print(f"RGB: {results['rgb']['conversion']['train_images']} train, {results['rgb']['conversion']['val_images']} val")
    print(f"Thermal: {results['thermal']['conversion']['train_images']} train, {results['thermal']['conversion']['val_images']} val")
    return results


def run_training(modality: str = "all"):
    """Step 2: Train models."""
    print("\n" + "="*60)
    print(f"STEP 2: TRAINING ({modality.upper()})")
    print("="*60)

    train_all(modality)
    print("Training complete!")


def run_evaluation(modality: str = "all"):
    """Step 3: Evaluate models."""
    print("\n" + "="*60)
    print(f"STEP 3: EVALUATION ({modality.upper()})")
    print("="*60)

    results = evaluate_all(modality)
    print("Evaluation complete!")
    return results


def run_export(modality: str = "all", formats: list = None, create_package: bool = True):
    """Step 4: Export models and create deployment packages."""
    print("\n" + "="*60)
    print(f"STEP 4: EXPORT ({modality.upper()})")
    print("="*60)

    if formats is None:
        formats = EXPORT_FORMATS

    from training.configs.config import OUTPUT_RGB, OUTPUT_THERMAL, OUTPUT_FUSION, MODEL_NAME

    exported = {}

    if modality in ("rgb", "all"):
        try:
            best = OUTPUT_RGB / MODEL_NAME / "train" / "weights" / "best.pt"
            if best.exists():
                out_dir = OUTPUT_RGB / MODEL_NAME / "exports"
                exported["rgb"] = export_model(best, out_dir, formats, EXPORT_IMG_SIZE)
                if create_package:
                    pkg = out_dir / f"{MODEL_NAME}_rgb_deployment.zip"
                    create_deployment_package(best.parent, pkg)
                    download_from_colab(pkg)
        except Exception as e:
            print(f"RGB export error: {e}")

    if modality in ("thermal", "all"):
        try:
            best = OUTPUT_THERMAL / MODEL_NAME / "train" / "weights" / "best.pt"
            if best.exists():
                out_dir = OUTPUT_THERMAL / MODEL_NAME / "exports"
                exported["thermal"] = export_model(best, out_dir, formats, EXPORT_IMG_SIZE)
                if create_package:
                    pkg = out_dir / f"{MODEL_NAME}_thermal_deployment.zip"
                    create_deployment_package(best.parent, pkg)
                    download_from_colab(pkg)
        except Exception as e:
            print(f"Thermal export error: {e}")

    if modality in ("fusion", "all"):
        try:
            best = OUTPUT_FUSION / MODEL_NAME / "train" / "weights" / "best.pt"
            if best.exists():
                out_dir = OUTPUT_FUSION / MODEL_NAME / "exports"
                exported["fusion"] = export_model(best, out_dir, formats, EXPORT_IMG_SIZE)
                if create_package:
                    pkg = out_dir / f"{MODEL_NAME}_fusion_deployment.zip"
                    create_deployment_package(best.parent, pkg)
                    download_from_colab(pkg)
        except Exception as e:
            print(f"Fusion export error: {e}")

    print("Export complete!")
    return exported


def run_full_pipeline(modality: str = "all", skip_preprocessing: bool = False,
                      skip_training: bool = False, skip_evaluation: bool = False,
                      skip_export: bool = False):
    """Run the complete pipeline."""
    print("\n" + "#"*60)
    print("# MULTIMODAL RGB-THERMAL TRAINING PIPELINE")
    print("#"*60)
    print(f"Modality: {modality}")
    print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, Size: {IMG_SIZE}")
    print(f"Model: {MODEL_NAME}, Seed: {SEED}")
    print(f"Device: {DEVICE}")
    print("#"*60)

    if not skip_preprocessing:
        run_preprocessing()
    else:
        print("\n>>> Skipping preprocessing")

    if not skip_training:
        run_training(modality)
    else:
        print("\n>>> Skipping training")

    if not skip_evaluation:
        run_evaluation(modality)
    else:
        print("\n>>> Skipping evaluation")

    if not skip_export:
        run_export(modality)
    else:
        print("\n>>> Skipping export")

    print("\n" + "#"*60)
    print("# PIPELINE COMPLETE")
    print("#"*60)
    print(f"All outputs in: {WORK_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multimodal RGB-Thermal Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m training.main                    # Run full pipeline (all modalities)
  python -m training.main --modality rgb     # Train only RGB
  python -m training.main --skip-training    # Only preprocess and evaluate existing models
  python -m training.main --only preprocessing --modality all
        """
    )
    parser.add_argument("--modality", type=str, default="all",
                        choices=["rgb", "thermal", "fusion", "all"],
                        help="Which modality to train/evaluate")
    parser.add_argument("--skip-preprocessing", action="store_true",
                        help="Skip VOC->YOLO conversion")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training (use existing weights)")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Skip evaluation")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip model export")
    parser.add_argument("--only", type=str,
                        choices=["preprocessing", "training", "evaluation", "export"],
                        help="Run only this step")

    args = parser.parse_args()

    if args.only:
        if args.only == "preprocessing":
            run_preprocessing()
        elif args.only == "training":
            run_training(args.modality)
        elif args.only == "evaluation":
            run_evaluation(args.modality)
        elif args.only == "export":
            run_export(args.modality)
    else:
        run_full_pipeline(
            modality=args.modality,
            skip_preprocessing=args.skip_preprocessing,
            skip_training=args.skip_training,
            skip_evaluation=args.skip_evaluation,
            skip_export=args.skip_export
        )