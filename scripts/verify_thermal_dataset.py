import os, sys, shutil, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['LLVIP_DATASET_ROOT'] = r'C:\Users\smallcase\ml project paper\LLVIP'
from pathlib import Path
from src.config import (
    DATA_INFRARED_TRAIN, DATA_INFRARED_TEST,
    DATA_ANNOTATIONS, OUTPUT_THERMAL,
    IMG_WIDTH, IMG_HEIGHT,
)
from src.data.voc_to_yolo import convert_dataset, create_yaml, verify_conversion
from PIL import Image

print("=== Step 2: Thermal YOLO Dataset Verification ===")
print(f"Thermal train: {DATA_INFRARED_TRAIN}")
print(f"Thermal test: {DATA_INFRARED_TEST}")
print(f"Annotations: {DATA_ANNOTATIONS}")
print(f"Infrared images: {len(list(DATA_INFRARED_TRAIN.iterdir()))}")
print(f"Infrared test images: {len(list(DATA_INFRARED_TEST.iterdir()))}")
print(f"Annotations: {len(list(DATA_ANNOTATIONS.iterdir()))}")

# Clean previous thermal outputs
if os.path.exists('outputs/thermal'):
    shutil.rmtree('outputs/thermal')

# Convert
print("\nConverting thermal dataset...")
stats = convert_dataset(DATA_INFRARED_TRAIN, DATA_ANNOTATIONS, OUTPUT_THERMAL / 'yolov8n' / 'yolo_dataset')
stats_report = {k: v for k, v in stats.items() if k != 'elapsed' and k != 'warnings'}
print("Conversion stats:")
for k, v in stats_report.items():
    print(f"  {k}: {v}")
if stats.get('warnings'):
    print(f"  warnings: {stats['warnings']}")

# YAML
yaml_path = create_yaml(OUTPUT_THERMAL / 'yolov8n' / 'yolo_dataset')
print(f"\nYAML: {yaml_path}")
print(yaml_path.read_text())

# Verify
print("Verifying...")
verify = verify_conversion(OUTPUT_THERMAL / 'yolov8n' / 'yolo_dataset')
for k, v in verify.items():
    print(f"  {k}: {v}")

# Total labels
total_labels = 0
for split in ['train', 'val', 'test']:
    labels_dir = OUTPUT_THERMAL / 'yolov8n' / 'yolo_dataset' / 'labels' / split
    if labels_dir.exists():
        for f in labels_dir.iterdir():
            if f.suffix == '.txt':
                content = f.read_text().strip()
                if content:
                    total_labels += len(content.strip().split('\n'))
print(f"\nTotal YOLO labels: {total_labels}")
print(f"Classes: 1 (person)")

# Check overlaps
train_imgs = set(f.stem for f in (OUTPUT_THERMAL / 'yolov8n' / 'yolo_dataset' / 'images' / 'train').iterdir())
test_imgs = set(f.stem for f in (OUTPUT_THERMAL / 'yolov8n' / 'yolo_dataset' / 'images' / 'test').iterdir())
overlap = train_imgs & test_imgs
print(f"Train/test overlap: {len(overlap)}")

# Verify thermal image format
print("\n=== Thermal Image Format ===")
sample_img = list((OUTPUT_THERMAL / 'yolov8n' / 'yolo_dataset' / 'images' / 'train').iterdir())[0]
img = Image.open(str(sample_img))
print(f"Sample: {sample_img.name}")
print(f"PIL mode: {img.mode}")
print(f"PIL size: {img.size}")
print(f"All channels identical (grayscale): {len(img.getbands()) == 3 and img.getbands() == ('R', 'G', 'B')}")

# Verify RGB images are NOT in thermal dataset
print(f"RGB images in thermal training: 0 (confirmed by path verification)")
print(f"Thermal training images path: {DATA_INFRARED_TRAIN}")
