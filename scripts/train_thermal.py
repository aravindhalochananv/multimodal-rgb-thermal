import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['LLVIP_DATASET_ROOT'] = r'C:\Users\smallcase\ml project paper\LLVIP'
from src.config import OUTPUT_THERMAL
from src.training.train_yolo import run

print("=== Step 5: Train Thermal-Only Model ===")
print("Configuration:")
print("  Model: yolov8n (COCO pretrained)")
print("  Image size: 320x320")
print("  Batch size: 4")
print("  Epochs: 2")
print("  Device: CPU")
print("  Seed: 42")
print()

result = run(
    modality="thermal",
    model_name="yolov8n",
    epochs=2,
    batch_size=4,
    img_size=320,
    patience=10,
    force_rebuild=False,
)
print(f"\nTraining result: {result}")
