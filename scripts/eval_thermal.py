import os, sys, json, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('LLVIP_DATASET_ROOT', str(Path(__file__).resolve().parents[1] / 'data' / 'LLVIP'))
from pathlib import Path
from src.config import OUTPUT_THERMAL
from src.evaluation.evaluate_yolo import evaluate

print("=== Step 6: Evaluate Thermal Model ===")
results = evaluate(modality="thermal", model_name="yolov8n", img_size=320, batch_size=4)

if results:
    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    output_path = OUTPUT_THERMAL / "thermal_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    csv_path = OUTPUT_THERMAL / "thermal_results.csv"
    flat = {k: v for k, v in results.items() if not isinstance(v, dict)}
    if results.get("inference_time"):
        flat.update(results["inference_time"])
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat.keys())
        writer.writeheader()
        writer.writerow(flat)
    print(f"CSV results saved to: {csv_path}")
else:
    print("Evaluation failed - no results")
