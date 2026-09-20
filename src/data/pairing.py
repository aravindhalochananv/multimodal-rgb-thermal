"""RGB-Thermal pairing logic for LLVIP dataset.

Discovers paired RGB/visible, thermal/infrared, and annotation files
using filename stem matching. Detects and reports missing members
rather than silently dropping them.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class PairRecord:
    id: str
    rgb_path: Path
    thermal_path: Path
    annotation_path: Path
    split: str


@dataclass
class PairReport:
    total_rgb: int = 0
    total_thermal: int = 0
    total_annotations: int = 0
    valid_pairs: int = 0
    missing_thermal: int = 0
    missing_rgb: int = 0
    missing_annotations: int = 0
    duplicate_rgb: int = 0
    duplicate_thermal: int = 0
    duplicate_pairs: int = 0
    orphan_annotations: int = 0
    images_without_annotations: int = 0
    invalid_pairs: list = field(default_factory=list)
    valid_pair_records: list = field(default_factory=list)


def _discover_files(directory: Path, extension: str) -> dict[str, Path]:
    if not directory.exists():
        return {}
    return {f.stem: f for f in sorted(directory.rglob(f"*{extension}")) if f.is_file()}


def create_pair_records(
    rgb_dir: Path,
    thermal_dir: Path,
    annotation_dir: Path,
    rgb_ext: str = ".jpg",
    thermal_ext: str = ".jpg",
    annotation_ext: str = ".xml",
) -> tuple[list[PairRecord], PairReport]:
    rgb_files = _discover_files(rgb_dir, rgb_ext)
    thermal_files = _discover_files(thermal_dir, thermal_ext)
    annotation_files = _discover_files(annotation_dir, annotation_ext)

    report = PairReport(
        total_rgb=len(rgb_files),
        total_thermal=len(thermal_files),
        total_annotations=len(annotation_files),
    )

    all_stems = set(rgb_files.keys()) | set(thermal_files.keys()) | set(annotation_files.keys())

    rgb_only = set(rgb_files.keys()) - set(thermal_files.keys())
    thermal_only = set(thermal_files.keys()) - set(rgb_files.keys())
    annotation_only = set(annotation_files.keys()) - set(rgb_files.keys())
    image_no_annotation = set(rgb_files.keys()) - set(annotation_files.keys())

    report.missing_thermal = len(rgb_only)
    report.missing_rgb = len(thermal_only)
    report.missing_annotations = len(annotation_only)
    report.orphan_annotations = len(annotation_only)
    report.images_without_annotations = len(image_no_annotation)

    report.invalid_pairs = []
    if rgb_only:
        report.invalid_pairs.append({
            "type": "missing_thermal",
            "count": len(rgb_only),
            "stems": sorted(rgb_only)[:100],
        })
    if thermal_only:
        report.invalid_pairs.append({
            "type": "missing_rgb",
            "count": len(thermal_only),
            "stems": sorted(thermal_only)[:100],
        })
    if annotation_only:
        report.invalid_pairs.append({
            "type": "missing_annotation",
            "count": len(annotation_only),
            "stems": sorted(annotation_only)[:100],
        })

    records: list[PairRecord] = []
    seen_pairs: set[tuple] = set()
    duplicate_count = 0

    for stem in sorted(all_stems):
        rgb_path = rgb_files.get(stem)
        thermal_path = thermal_files.get(stem)
        annotation_path = annotation_files.get(stem)

        if rgb_path is None or thermal_path is None or annotation_path is None:
            continue

        pair_key = (str(rgb_path), str(thermal_path))
        if pair_key in seen_pairs:
            duplicate_count += 1
            continue
        seen_pairs.add(pair_key)

        split = "train" if "train" in str(rgb_path) else "test" if "test" in str(rgb_path) else "unknown"

        record = PairRecord(
            id=stem,
            rgb_path=rgb_path,
            thermal_path=thermal_path,
            annotation_path=annotation_path,
            split=split,
        )
        records.append(record)

    report.valid_pairs = len(records)
    report.duplicate_pairs = duplicate_count
    report.valid_pair_records = records

    return records, report
