import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['LLVIP_DATASET_ROOT'] = r'C:\Users\smallcase\ml project paper\LLVIP'
from pathlib import Path

print("=== Step 9: RGB vs Thermal Comparison ===\n")

# Load results
rgb_path = Path("outputs/rgb/rgb_results.json")
thermal_path = Path("outputs/thermal/thermal_results.json")

rgb = json.loads(rgb_path.read_text()) if rgb_path.exists() else {}
thermal = json.loads(thermal_path.read_text()) if thermal_path.exists() else {}

print(f"{'Metric':<20} {'RGB-only':>15} {'Thermal-only':>15} {'Delta':>10}")
print("-" * 65)

metrics = ['precision', 'recall', 'f1_score', 'mAP50', 'mAP50_95']
for m in metrics:
    rgb_val = rgb.get(m, 'N/A')
    thermal_val = thermal.get(m, 'N/A')
    if isinstance(rgb_val, (int, float)) and isinstance(thermal_val, (int, float)):
        delta = thermal_val - rgb_val
        print(f"{m:<20} {rgb_val:>15.4f} {thermal_val:>15.4f} {delta:>+10.4f}")
    else:
        print(f"{m:<20} {str(rgb_val):>15} {str(thermal_val):>15} {'N/A':>10}")

# Inference time
if rgb.get('inference_time') and thermal.get('inference_time'):
    rgb_inf = rgb['inference_time'].get('avg_per_image_ms', 'N/A')
    thermal_inf = thermal['inference_time'].get('avg_per_image_ms', 'N/A')
    if isinstance(rgb_inf, (int, float)) and isinstance(thermal_inf, (int, float)):
        delta = thermal_inf - rgb_inf
        print(f"{'inference_time_ms':<20} {rgb_inf:>15.2f} {thermal_inf:>15.2f} {delta:>+10.2f}")

print("\nNOTE: Thermal-only experiments is in progress.")
print("Comparison between modalities is descriptive only.")
print("No conclusion about superiority is drawn.")

# Save comparison
output_dir = Path("outputs/comparison")
output_dir.mkdir(parents=True, exist_ok=True)

comparison = {
    "rgb": {m: rgb.get(m) for m in metrics},
    "thermal": {m: thermal.get(m) for m in metrics},
}
if rgb.get('inference_time'):
    comparison['rgb']['inference_time_ms'] = rgb['inference_time'].get('avg_per_image_ms')
if thermal.get('inference_time'):
    comparison['thermal']['inference_time_ms'] = thermal['inference_time'].get('avg_per_image_ms')

cmp_path = output_dir / "rgb_thermal_comparison.json"
with open(cmp_path, "w") as f:
    json.dump(comparison, f, indent=2, default=str)
print(f"\nComparison saved to: {cmp_path}")
