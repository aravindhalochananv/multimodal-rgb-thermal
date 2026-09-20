"""Comparison script for all YOLO experiments.

Reads test results from all experiments and creates a comparison table.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.config import OUTPUT_RGB, OUTPUT_THERMAL


def collect_results() -> list:
    """Collect metrics from all experiment result files."""
    experiments = [
        ("rgb", "yolov8n"), ("rgb", "yolov8s"),
        ("thermal", "yolov8n"), ("thermal", "yolov8s"),
    ]

    all_results = []
    for modality, model in experiments:
        base = (OUTPUT_RGB if modality == "rgb" else OUTPUT_THERMAL) / model
        results_file = base / "test_results.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
            data["modality"] = modality
            data["model"] = model
            all_results.append(data)
        else:
            all_results.append({
                "modality": modality, "model": model,
                "status": "not_evaluated",
            })

    return all_results


def create_comparison_table(results: list, output_path: Path):
    """Create a formatted comparison CSV table."""
    rows = []
    for r in results:
        if r.get("status") == "not_evaluated":
            rows.append({
                "Modality": r["modality"].capitalize(),
                "Model": r["model"],
                "Precision": "N/A",
                "Recall": "N/A",
                "F1": "N/A",
                "mAP@50": "N/A",
                "mAP@50:95": "N/A",
                "Inference Time (ms)": "N/A",
            })
        else:
            inf_time = r.get("inference_time", {})
            if isinstance(inf_time, dict):
                inf_str = f"{inf_time.get('avg_per_image_ms', 'N/A')} ms/img"
            else:
                inf_str = str(inf_time)

            rows.append({
                "Modality": r["modality"].capitalize(),
                "Model": r["model"],
                "Precision": r.get("precision", "N/A"),
                "Recall": r.get("recall", "N/A"),
                "F1": r.get("f1_score", "N/A"),
                "mAP@50": r.get("mAP50", "N/A"),
                "mAP@50:95": r.get("mAP50_95", "N/A"),
                "Inference Time (ms)": inf_str,
            })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Modality", "Model", "Precision", "Recall", "F1", "mAP@50", "mAP@50:95", "Inference Time (ms)"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("\nComparison Table:")
    for row in rows:
        print(f"  {row['Modality']:10s} {row['Model']:10s} "
              f"P={row['Precision']} R={row['Recall']} "
              f"F1={row['F1']} mAP50={row['mAP@50']} "
              f"mAP50:95={row['mAP@50:95']} {row['Inference Time (ms)']}")

    return rows


def save_json(results: list, output_path: Path):
    """Save raw results as JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def main():
    output_dir = OUTPUT_RGB.parent / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = collect_results()

    table_path = output_dir / "comparison.csv"
    json_path = output_dir / "comparison.json"
    summary_path = output_dir / "comparison.txt"

    rows = create_comparison_table(results, table_path)
    save_json(results, json_path)

    with open(summary_path, "w") as f:
        f.write("RGB vs Thermal YOLO Experiment Comparison\n")
        f.write("=" * 60 + "\n")
        for row in rows:
            f.write(f"{row}\n")

    print(f"\nComparison saved to: {table_path}")
    print(f"JSON saved to: {json_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
