"""
Dataset copy/verification utility.
Copies LLVIP dataset to training workspace for portability.
"""

import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.configs.config import (
    DATA_ROOT, WORK_DIR,
    VISIBLE_TRAIN, VISIBLE_TEST,
    INFRARED_TRAIN, INFRARED_TEST,
    ANNOTATIONS,
    NUM_VISIBLE_TRAIN, NUM_VISIBLE_TEST,
    NUM_INFRARED_TRAIN, NUM_INFRARED_TEST,
    NUM_ANNOTATIONS
)


def verify_dataset(root: Path) -> bool:
    """Verify LLVIP dataset structure and counts."""
    dirs = {
        "visible/train": root / "visible" / "train",
        "visible/test": root / "visible" / "test",
        "infrared/train": root / "infrared" / "train",
        "infrared/test": root / "infrared" / "test",
        "Annotations": root / "Annotations",
    }

    expected = {
        "visible/train": NUM_VISIBLE_TRAIN,
        "visible/test": NUM_VISIBLE_TEST,
        "infrared/train": NUM_INFRARED_TRAIN,
        "infrared/test": NUM_INFRARED_TEST,
        "Annotations": NUM_ANNOTATIONS,
    }

    print(f"Verifying dataset at: {root}")
    all_ok = True

    for name, path in dirs.items():
        if not path.exists():
            print(f"  ❌ MISSING: {name} -> {path}")
            all_ok = False
            continue

        count = len(list(path.iterdir()))
        exp = expected[name]
        status = "✅" if count == exp else "⚠️"
        print(f"  {status} {name}: {count} (expected {exp})")
        if count != exp:
            all_ok = False

    # Check pairing
    rgb_stems = {f.stem for f in dirs["visible/train"].iterdir()} | {f.stem for f in dirs["visible/test"].iterdir()}
    thermal_stems = {f.stem for f in dirs["infrared/train"].iterdir()} | {f.stem for f in dirs["infrared/test"].iterdir()}
    ann_stems = {f.stem for f in dirs["Annotations"].iterdir()}

    paired = rgb_stems & thermal_stems & ann_stems
    print(f"\n  Paired samples: {len(paired)} / {len(rgb_stems)} RGB, {len(thermal_stems)} thermal, {len(ann_stems)} annotations")

    if len(paired) != NUM_ANNOTATIONS:
        print(f"  ⚠️ Pairing mismatch!")
        all_ok = False

    return all_ok


def copy_dataset_to_workspace(src_root: Path, dst_root: Path) -> None:
    """Copy entire LLVIP dataset to workspace for portability."""
    dst_root = Path(dst_root)
    dirs = [
        ("visible/train", "visible/train"),
        ("visible/test", "visible/test"),
        ("infrared/train", "infrared/train"),
        ("infrared/test", "infrared/test"),
        ("Annotations", "Annotations"),
    ]

    print(f"Copying dataset from {src_root} to {dst_root}...")
    for src_rel, dst_rel in dirs:
        src = src_root / src_rel
        dst = dst_root / dst_rel
        if src.exists():
            print(f"  Copying {src_rel}...")
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            print(f"  ⚠️ Missing: {src_rel}")

    print("Copy complete!")


def create_dataset_symlinks(src_root: Path, dst_root: Path) -> None:
    """Create symlinks instead of copying (saves space)."""
    dst_root = Path(dst_root)
    dirs = [
        "visible/train", "visible/test",
        "infrared/train", "infrared/test",
        "Annotations"
    ]

    print(f"Creating symlinks from {src_root} to {dst_root}...")
    for d in dirs:
        src = src_root / d
        dst = dst_root / d
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(src, target_is_directory=True)
            print(f"  Linked {d}")
        else:
            print(f"  Exists: {d}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify or copy LLVIP dataset")
    parser.add_argument("--verify", action="store_true", help="Verify dataset structure")
    parser.add_argument("--copy", action="store_true", help="Copy dataset to workspace")
    parser.add_argument("--symlink", action="store_true", help="Create symlinks to dataset")
    parser.add_argument("--src", type=str, default=str(DATA_ROOT), help="Source dataset root")
    parser.add_argument("--dst", type=str, default=str(WORK_DIR / "LLVIP"), help="Destination root")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if args.verify or not (args.copy or args.symlink):
        ok = verify_dataset(src)
        sys.exit(0 if ok else 1)

    if args.copy:
        copy_dataset_to_workspace(src, dst)
    elif args.symlink:
        create_dataset_symlinks(src, dst)