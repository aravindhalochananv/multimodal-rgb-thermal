"""Dataset audit pipeline for LLVIP dataset.

Performs comprehensive verification of RGB/thermal images,
annotations, pairing, and data quality.
"""

from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import json
import csv
import io
import sys
import time
from collections import Counter

import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    DATA_VISIBLE_ROOT, DATA_VISIBLE_TRAIN, DATA_VISIBLE_TEST,
    DATA_INFRARED_ROOT, DATA_INFRARED_TRAIN, DATA_INFRARED_TEST,
    DATA_ANNOTATIONS,
    NUM_VISIBLE_TRAIN, NUM_VISIBLE_TEST,
    NUM_INFRARED_TRAIN, NUM_INFRARED_TEST,
    NUM_TOTAL_PAIRS, NUM_ANNOTATIONS,
)
from src.data.pairing import create_pair_records, PairReport
from src.data.dataset_loader import get_image_info
from src.data.preprocessing import parse_voc_annotation


@dataclass
class AuditResult:
    check_name: str
    status: str
    value: Any = None
    details: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class DatasetAudit:
    def __init__(self, output_dir: Optional[Path] = None):
        from src.config import DATA_AUDIT
        self.output_dir = output_dir or DATA_AUDIT
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[AuditResult] = []
        self.pair_report: Optional[PairReport] = None
        self.pair_records: list = []

    def _add_result(self, check_name: str, status: str, value: Any = None, details: Optional[str] = None):
        self.results.append(AuditResult(
            check_name=check_name,
            status=status,
            value=value,
            details=details,
        ))

    def run_full_audit(self) -> dict:
        audit_log = []
        audit_log.append("=" * 60)
        audit_log.append("LLVIP DATASET AUDIT REPORT")
        audit_log.append("=" * 60)
        audit_log.append(f"Dataset root: {DATA_VISIBLE_ROOT.parent}")
        audit_log.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        audit_log.append("")

        self._audit_file_counts(audit_log)
        self._audit_pairing(audit_log)
        self._audit_image_properties(audit_log)
        self._audit_corrupt_images(audit_log)
        self._audit_annotations(audit_log)
        self._audit_bounding_boxes(audit_log)
        self._audit_train_test_integrity(audit_log)
        self._audit_class_distribution(audit_log)

        summary = self._build_summary()
        audit_log.append("")
        audit_log.append("=" * 60)
        audit_log.append("AUDIT SUMMARY")
        audit_log.append("=" * 60)
        for k, v in summary.items():
            audit_log.append(f"  {k}: {v}")

        audit_text = "\n".join(audit_log)
        print(audit_text)

        self._save_results(audit_log, summary)

        return {
            "audit_text": audit_text,
            "summary": summary,
            "results": [asdict(r) for r in self.results],
        }

    def _audit_file_counts(self, log: list):
        log.append("\n--- File Count Verification ---")

        rgb_train_count = len([f for f in DATA_VISIBLE_TRAIN.rglob("*") if f.is_file()])
        rgb_test_count = len([f for f in DATA_VISIBLE_TEST.rglob("*") if f.is_file()])
        rgb_total = rgb_train_count + rgb_test_count
        thermal_train_count = len([f for f in DATA_INFRARED_TRAIN.rglob("*") if f.is_file()])
        thermal_test_count = len([f for f in DATA_INFRARED_TEST.rglob("*") if f.is_file()])
        thermal_total = thermal_train_count + thermal_test_count
        ann_count = len([f for f in DATA_ANNOTATIONS.rglob("*") if f.is_file()])

        self._add_result("RGB image count (train)", "PASS" if rgb_train_count == NUM_VISIBLE_TRAIN else "FAIL",
                         rgb_train_count, f"Expected {NUM_VISIBLE_TRAIN}")
        self._add_result("RGB image count (test)", "PASS" if rgb_test_count == NUM_VISIBLE_TEST else "FAIL",
                         rgb_test_count, f"Expected {NUM_VISIBLE_TEST}")
        self._add_result("RGB image count (total)", "PASS" if rgb_total == NUM_TOTAL_PAIRS else "FAIL",
                         rgb_total, f"Expected {NUM_TOTAL_PAIRS}")
        self._add_result("Thermal image count (train)", "PASS" if thermal_train_count == NUM_INFRARED_TRAIN else "FAIL",
                         thermal_train_count, f"Expected {NUM_INFRARED_TRAIN}")
        self._add_result("Thermal image count (test)", "PASS" if thermal_test_count == NUM_INFRARED_TEST else "FAIL",
                         thermal_test_count, f"Expected {NUM_INFRARED_TEST}")
        self._add_result("Thermal image count (total)", "PASS" if thermal_total == NUM_TOTAL_PAIRS else "FAIL",
                         thermal_total, f"Expected {NUM_TOTAL_PAIRS}")
        self._add_result("Annotation count", "PASS" if ann_count == NUM_ANNOTATIONS else "FAIL",
                         ann_count, f"Expected {NUM_ANNOTATIONS}")

        log.append(f"  RGB images: {rgb_total} (train={rgb_train_count}, test={rgb_test_count})")
        log.append(f"  Thermal images: {thermal_total} (train={thermal_train_count}, test={thermal_test_count})")
        log.append(f"  Annotations: {ann_count}")

    def _audit_pairing(self, log: list):
        log.append("\n--- RGB/Thermal Pairing Verification ---")

        # Overall pairing check (all data)
        records_all, report_all = create_pair_records(
            DATA_VISIBLE_ROOT, DATA_INFRARED_ROOT, DATA_ANNOTATIONS,
        )

        # Train-level check
        records_train, report_train = create_pair_records(
            DATA_VISIBLE_TRAIN, DATA_INFRARED_TRAIN, DATA_ANNOTATIONS,
        )

        # Test-level check
        records_test, report_test = create_pair_records(
            DATA_VISIBLE_TEST, DATA_INFRARED_TEST, DATA_ANNOTATIONS,
        )

        self.pair_report = report_all
        self.pair_records = records_all

        all_valid = report_all.valid_pairs

        self._add_result("Valid RGB-Thermal pairs (total)", "PASS" if all_valid == NUM_TOTAL_PAIRS else "FAIL",
                         all_valid, f"Found {all_valid} paired records total")
        self._add_result("Valid RGB-Thermal pairs (train)", "PASS" if report_train.valid_pairs == NUM_VISIBLE_TRAIN else "FAIL",
                         report_train.valid_pairs, f"Found {report_train.valid_pairs} paired records in train")
        self._add_result("Valid RGB-Thermal pairs (test)", "PASS" if report_test.valid_pairs == NUM_VISIBLE_TEST else "FAIL",
                         report_test.valid_pairs, f"Found {report_test.valid_pairs} paired records in test")
        self._add_result("Missing thermal images", "PASS" if report_all.missing_thermal == 0 else "FAIL",
                         report_all.missing_thermal, f"{report_all.missing_thermal} RGB images without thermal")
        self._add_result("Missing RGB images", "PASS" if report_all.missing_rgb == 0 else "FAIL",
                         report_all.missing_rgb, f"{report_all.missing_rgb} thermal images without RGB")
        self._add_result("Missing annotations (overall)", "PASS" if report_all.missing_annotations == 0 else "FAIL",
                         report_all.missing_annotations, f"{report_all.missing_annotations} images without annotations")
        self._add_result("Duplicate RGB files", "PASS" if report_all.duplicate_rgb == 0 else "WARN",
                         report_all.duplicate_rgb, "Check filename uniqueness within modality")
        self._add_result("Duplicate thermal files", "PASS" if report_all.duplicate_thermal == 0 else "WARN",
                         report_all.duplicate_thermal, "Check filename uniqueness within modality")
        self._add_result("Duplicate RGB-thermal pairs", "PASS" if report_all.duplicate_pairs == 0 else "WARN",
                         report_all.duplicate_pairs, "Check for duplicate cross-modal pairs")

        log.append(f"  Valid pairs (total): {all_valid}")
        log.append(f"  Valid pairs (train): {report_train.valid_pairs}")
        log.append(f"  Valid pairs (test): {report_test.valid_pairs}")
        log.append(f"  Missing thermal: {report_all.missing_thermal}")
        log.append(f"  Missing RGB: {report_all.missing_rgb}")
        log.append(f"  Missing annotations: {report_all.missing_annotations}")
        log.append(f"  Duplicate pairs: {report_all.duplicate_pairs}")

        if report_all.invalid_pairs:
            log.append("  Issues found:")
            for inv in report_all.invalid_pairs:
                log.append(f"    {inv['type']}: {inv['count']} items")
                if inv['stems']:
                    log.append(f"      Sample: {inv['stems'][:5]}")

    def _audit_image_properties(self, log: list):
        log.append("\n--- Image Properties ---")

        dimensions_rgb = []
        dimensions_thermal = []
        formats_rgb = Counter()
        formats_thermal = Counter()
        corrupt_count = 0
        corrupt_images = []

        splits = [("RGB Train", DATA_VISIBLE_TRAIN), ("RGB Test", DATA_VISIBLE_TEST),
                   ("Thermal Train", DATA_INFRARED_TRAIN), ("Thermal Test", DATA_INFRARED_TEST)]

        for label, directory in splits:
            sample_size = 500 if directory.exists() else 0
            files = sorted([f for f in directory.rglob("*") if f.is_file()])[:sample_size]
            for f in files:
                info = get_image_info(f)
                if info is None:
                    continue
                if info["corrupt"]:
                    corrupt_count += 1
                    corrupt_images.append(str(f))
                    continue
                if "visible" in str(directory):
                    dimensions_rgb.append((info["width"], info["height"]))
                    formats_rgb[info["format"]] += 1
                else:
                    dimensions_thermal.append((info["width"], info["height"]))
                    formats_thermal[info["format"]] += 1

        self._add_result("Corrupt/unreadable images (sampled)", "PASS" if corrupt_count == 0 else "FAIL",
                         corrupt_count, f"Found {corrupt_count} corrupt images in sample" if corrupt_count > 0 else "No corrupt images found in sample")

        if dimensions_rgb:
            unique_rgb_dims = set(dimensions_rgb)
            self._add_result("RGB dimensions consistent", "PASS" if len(unique_rgb_dims) == 1 else "WARN",
                             str(unique_rgb_dims), f"Found {len(unique_rgb_dims)} unique dimensions")
            log.append(f"  RGB dimensions: {unique_rgb_dims}")

        if dimensions_thermal:
            unique_thermal_dims = set(dimensions_thermal)
            self._add_result("Thermal dimensions consistent", "PASS" if len(unique_thermal_dims) == 1 else "WARN",
                             str(unique_thermal_dims), f"Found {len(unique_thermal_dims)} unique dimensions")
            log.append(f"  Thermal dimensions: {unique_thermal_dims}")

        self._add_result("RGB image format", "PASS" if len(formats_rgb) <= 1 else "INFO",
                         dict(formats_rgb), "Image formats found in RGB set")
        self._add_result("Thermal image format", "PASS" if len(formats_thermal) <= 1 else "INFO",
                         dict(formats_thermal), "Image formats found in thermal set")
        log.append(f"  RGB formats: {dict(formats_rgb)}")
        log.append(f"  Thermal formats: {dict(formats_thermal)}")

    def _audit_corrupt_images(self, log: list):
        log.append("\n--- Corrupt Image Check (Full Scan) ---")
        corrupt = []
        total = 0
        for directory in [DATA_VISIBLE_TRAIN, DATA_VISIBLE_TEST, DATA_INFRARED_TRAIN, DATA_INFRARED_TEST]:
            if not directory.exists():
                continue
            files = sorted([f for f in directory.rglob("*") if f.is_file()])
            batch_size = 1000
            for i in range(0, len(files), batch_size):
                batch = files[i:i+batch_size]
                for f in batch:
                    total += 1
                    try:
                        img = Image.open(f)
                        img.verify()
                    except Exception:
                        corrupt.append(str(f))
                if total % 5000 == 0:
                    log.append(f"  Checked {total} images... ({len(corrupt)} corrupt)")

        self._add_result("Corrupt images (full scan)", "PASS" if len(corrupt) == 0 else "FAIL",
                         len(corrupt), f"Checked {total} images, {len(corrupt)} corrupt")
        log.append(f"  Total checked: {total}")
        log.append(f"  Corrupt: {len(corrupt)}")
        if corrupt:
            log.append(f"  Sample corrupt: {corrupt[:5]}")

    def _audit_annotations(self, log: list):
        log.append("\n--- Annotation Analysis ---")

        ann_files = sorted([f for f in DATA_ANNOTATIONS.rglob("*") if f.is_file()])
        format_types = Counter()
        annotations_without_image = []
        empty_annotations = []
        all_classes = Counter()
        annotation_stems = set()
        image_stems = set()

        rgb_stems = set()
        for d in [DATA_VISIBLE_TRAIN, DATA_VISIBLE_TEST]:
            if d.exists():
                for f in d.rglob("*"):
                    if f.is_file():
                        rgb_stems.add(f.stem)

        for af in ann_files:
            stem = af.stem
            annotation_stems.add(stem)
            format_types[af.suffix.lower()] += 1

            if stem not in rgb_stems:
                annotations_without_image.append(stem)

            try:
                info = parse_voc_annotation(af)
                if not info["objects"]:
                    empty_annotations.append(stem)
                for obj in info["objects"]:
                    all_classes[obj["name"]] += 1
            except Exception:
                pass

        self._add_result("Annotation format", "PASS" if len(format_types) == 1 else "INFO",
                         dict(format_types), f"Annotation formats: {dict(format_types)}")
        self._add_result("Annotations without corresponding image", "PASS" if len(annotations_without_image) == 0 else "FAIL",
                         len(annotations_without_image), f"{len(annotations_without_image)} orphan annotations")
        self._add_result("Empty annotations (no objects)", "PASS" if len(empty_annotations) == 0 else "WARN",
                         len(empty_annotations), f"{len(empty_annotations)} annotations with no objects")
        self._add_result("Annotation class labels", "INFO", dict(all_classes), "Classes found in annotations")
        log.append(f"  Format types: {dict(format_types)}")
        log.append(f"  Orphan annotations: {len(annotations_without_image)}")
        log.append(f"  Empty annotations: {len(empty_annotations)}")
        log.append(f"  Classes: {dict(all_classes)}")
        if empty_annotations:
            log.append(f"  Sample empty: {empty_annotations[:10]}")

    def _audit_bounding_boxes(self, log: list):
        log.append("\n--- Bounding Box Validation ---")

        invalid_bbox_count = 0
        out_of_bounds_count = 0
        invalid_examples = []
        oob_examples = []

        ann_files = sorted([f for f in DATA_ANNOTATIONS.rglob("*") if f.is_file()])
        sample_size = min(len(ann_files), 3000)

        for af in ann_files[:sample_size]:
            try:
                info = parse_voc_annotation(af)
                img_w = info["width"]
                img_h = info["height"]
                for obj in info["objects"]:
                    bbox = obj["bbox"]
                    xmin, ymin, xmax, ymax = bbox

                    if xmin >= xmax or ymin >= ymax:
                        invalid_bbox_count += 1
                        if len(invalid_examples) < 10:
                            invalid_examples.append({
                                "file": str(af),
                                "bbox": bbox,
                                "reason": "xmin >= xmax or ymin >= ymax",
                            })

                    if xmin < 0 or ymin < 0 or xmax > img_w or ymax > img_h:
                        out_of_bounds_count += 1
                        if len(oob_examples) < 10:
                            oob_examples.append({
                                "file": str(af),
                                "bbox": bbox,
                                "image_size": (img_w, img_h),
                            })
            except Exception:
                pass

        self._add_result("Invalid bounding boxes (xmin>=xmax or ymin>=ymax)", "PASS" if invalid_bbox_count == 0 else "FAIL",
                         invalid_bbox_count, f"Found {invalid_bbox_count} invalid bboxes in sample of {sample_size}")
        self._add_result("Bounding boxes outside image boundaries", "PASS" if out_of_bounds_count == 0 else "FAIL",
                         out_of_bounds_count, f"Found {out_of_bounds_count} out-of-bounds bboxes in sample of {sample_size}")

        log.append(f"  Sample size: {sample_size}")
        log.append(f"  Invalid bboxes: {invalid_bbox_count}")
        log.append(f"  Out-of-bounds bboxes: {out_of_bounds_count}")
        if invalid_examples:
            log.append(f"  Invalid examples: {invalid_examples[:3]}")
        if oob_examples:
            log.append(f"  OOB examples: {oob_examples[:3]}")

    def _audit_train_test_integrity(self, log: list):
        log.append("\n--- Train/Test Integrity ---")

        rgb_train_stems = set(f.stem for f in DATA_VISIBLE_TRAIN.rglob("*") if f.is_file())
        rgb_test_stems = set(f.stem for f in DATA_VISIBLE_TEST.rglob("*") if f.is_file())
        thermal_train_stems = set(f.stem for f in DATA_INFRARED_TRAIN.rglob("*") if f.is_file())
        thermal_test_stems = set(f.stem for f in DATA_INFRARED_TEST.rglob("*") if f.is_file())

        train_overlap_rgb_thermal = rgb_train_stems & thermal_train_stems
        test_overlap_rgb_thermal = rgb_test_stems & thermal_test_stems
        train_overlap_count = len(train_overlap_rgb_thermal)
        test_overlap_count = len(test_overlap_rgb_thermal)

        filename_overlap = rgb_train_stems & rgb_test_stems
        thermal_overlap = thermal_train_stems & thermal_test_stems
        filename_overlap_count = len(filename_overlap)

        train_no_thermal = rgb_train_stems - thermal_train_stems
        test_no_thermal = rgb_test_stems - thermal_test_stems

        self._add_result("RGB-Thermal correspondence (train)", "PASS" if train_overlap_count == len(rgb_train_stems) else "FAIL",
                         f"{train_overlap_count}/{len(rgb_train_stems)}", "All RGB have matching thermal in train")
        self._add_result("RGB-Thermal correspondence (test)", "PASS" if test_overlap_count == len(rgb_test_stems) else "FAIL",
                         f"{test_overlap_count}/{len(rgb_test_stems)}", "All RGB have matching thermal in test")
        self._add_result("Train/test filename overlap (RGB)", "PASS" if filename_overlap_count == 0 else "FAIL",
                         filename_overlap_count, f"{filename_overlap_count} overlapping filenames between train and test")
        self._add_result("Train/test filename overlap (Thermal)", "PASS" if len(thermal_overlap) == 0 else "FAIL",
                         len(thermal_overlap), f"{len(thermal_overlap)} overlapping filenames between train and test")
        self._add_result("Missing thermal in train", "PASS" if len(train_no_thermal) == 0 else "FAIL",
                         len(train_no_thermal), f"{len(train_no_thermal)} RGB without thermal in train")
        self._add_result("Missing thermal in test", "PASS" if len(test_no_thermal) == 0 else "FAIL",
                         len(test_no_thermal), f"{len(test_no_thermal)} RGB without thermal in test")

        log.append(f"  RGB train stems: {len(rgb_train_stems)}")
        log.append(f"  RGB test stems: {len(rgb_test_stems)}")
        log.append(f"  RGB train/test overlap: {filename_overlap_count}")
        log.append(f"  Thermal train stems: {len(thermal_train_stems)}")
        log.append(f"  Thermal test stems: {len(thermal_test_stems)}")
        log.append(f"  Thermal train/test overlap: {len(thermal_overlap)}")
        log.append(f"  RGB-Thermal overlap (train): {train_overlap_count}")
        log.append(f"  RGB-Thermal overlap (test): {test_overlap_count}")

        if filename_overlap_count > 0:
            log.append(f"  WARNING: {filename_overlap_count} filenames appear in both train and test!")
            log.append(f"  Sample overlap: {list(filename_overlap)[:10]}")
            self._add_result("DATA LEAKAGE WARNING", "FAIL",
                             filename_overlap_count, "Filenames overlap between train and test")

    def _audit_class_distribution(self, log: list):
        log.append("\n--- Class Distribution ---")

        all_classes = Counter()
        ann_files = sorted([f for f in DATA_ANNOTATIONS.rglob("*") if f.is_file()])
        for af in ann_files:
            try:
                info = parse_voc_annotation(af)
                for obj in info["objects"]:
                    all_classes[obj["name"]] += 1
            except Exception:
                pass

        self._add_result("Annotation class distribution", "INFO", dict(all_classes), "Object classes in annotations")
        log.append(f"  Classes: {dict(all_classes)}")
        log.append(f"  Total objects: {sum(all_classes.values())}")

    def _build_summary(self) -> dict:
        results_dict = {r.check_name: {"status": r.status, "value": r.value} for r in self.results}
        return {
            "total_checks": len(self.results),
            "passed": sum(1 for r in self.results if r.status == "PASS"),
            "failed": sum(1 for r in self.results if r.status == "FAIL"),
            "warnings": sum(1 for r in self.results if r.status == "WARN"),
            "info": sum(1 for r in self.results if r.status == "INFO"),
            "results": results_dict,
        }

    def _save_results(self, audit_log: list, summary: dict):
        audit_path = self.output_dir / "dataset_audit.json"
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": summary,
                "results": [asdict(r) for r in self.results],
                "audit_log": audit_log,
            }, f, indent=2, default=str)

        csv_path = self.output_dir / "dataset_audit.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["check_name", "status", "value", "details", "timestamp"])
            for r in self.results:
                writer.writerow([r.check_name, r.status, str(r.value), r.details or "", r.timestamp])

        log_path = self.output_dir / "dataset_audit.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(audit_log))

        if self.pair_report:
            pairing_path = self.output_dir / "pairing_report.csv"
            with open(pairing_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["check", "count", "status", "details"])
                writer.writerow(["total_rgb", self.pair_report.total_rgb, "", ""])
                writer.writerow(["total_thermal", self.pair_report.total_thermal, "", ""])
                writer.writerow(["total_annotations", self.pair_report.total_annotations, "", ""])
                writer.writerow(["valid_pairs", self.pair_report.valid_pairs, "", ""])
                writer.writerow(["missing_thermal", self.pair_report.missing_thermal, "", ""])
                writer.writerow(["missing_rgb", self.pair_report.missing_rgb, "", ""])
                writer.writerow(["missing_annotations", self.pair_report.missing_annotations, "", ""])
                writer.writerow(["duplicate_pairs", self.pair_report.duplicate_pairs, "", ""])
                writer.writerow(["orphan_annotations", self.pair_report.orphan_annotations, "", ""])
                for inv in self.pair_report.invalid_pairs:
                    writer.writerow([inv["type"], inv["count"], "FAIL", str(inv["stems"][:10]) if inv["stems"] else ""])

        print(f"\nAudit results saved to: {self.output_dir}")
        print(f"  - dataset_audit.json")
        print(f"  - dataset_audit.csv")
        print(f"  - dataset_audit.txt")
        print(f"  - pairing_report.csv")
