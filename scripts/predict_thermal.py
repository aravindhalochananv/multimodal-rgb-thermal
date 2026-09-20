import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['LLVIP_DATASET_ROOT'] = r'C:\Users\smallcase\ml project paper\LLVIP'
from pathlib import Path
from ultralytics import YOLO
from src.config import DATA_INFRARED_TEST, OUTPUT_THERMAL

print("=== Step 7: Generate Thermal Predictions ===")
model = YOLO(str(OUTPUT_THERMAL / "yolov8n" / "train" / "weights" / "best.pt"))
output_dir = OUTPUT_THERMAL / "predictions"
output_dir.mkdir(parents=True, exist_ok=True)

test_images = sorted(DATA_INFRARED_TEST.iterdir())
sample = test_images[:10]

for img_path in sample:
    results = model.predict(str(img_path), imgsz=320, conf=0.25, verbose=False)
    result = results[0]
    out_path = output_dir / f"{img_path.stem}_pred.jpg"
    result.save(str(out_path))
    boxes = result.boxes
    print(f"\n{img_path.name}: {len(boxes)} predictions")
    for i, box in enumerate(boxes):
        conf = float(box.conf[0]) if len(box.conf) > 0 else 0
        cls = int(box.cls[0]) if len(box.cls) > 0 else -1
        cls_name = model.names.get(cls, "unknown")
        print(f"  Box {i}: class={cls_name}, conf={conf:.3f}")

print(f"\nPredictions saved to: {output_dir}")
print(f"Total prediction images: {len(list(output_dir.glob('*_pred.jpg')))}")
