import os
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

test_paths = [
    r'C:\Users\smallcase\ml project paper\LLVIP\infrared\test\190001.jpg',
    r'C:\Users\smallcase\ml project paper\LLVIP\infrared\test\190002.jpg',
    r'C:\Users\smallcase\ml project paper\LLVIP\infrared\train\010001.jpg',
    r'C:\Users\smallcase\ml project paper\LLVIP\visible\train\010001.jpg',
]

for p in test_paths:
    print(f"\n=== {Path(p).name} ===")
    img_pil = Image.open(p)
    print(f"PIL mode: {img_pil.mode}")
    print(f"PIL size: {img_pil.size}")
    if hasattr(img_pil, 'getbands'):
        print(f"PIL bands: {img_pil.getbands()}")

    img_cv = cv2.imread(p)
    print(f"OpenCV shape: {img_cv.shape}")
    print(f"OpenCV dtype: {img_cv.dtype}")

    if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
        b, g, r = img_cv[:, :, 0], img_cv[:, :, 1], img_cv[:, :, 2]
        print(f"B == G: {np.array_equal(b, g)}")
        print(f"G == R: {np.array_equal(g, r)}")
        print(f"B == R: {np.array_equal(b, r)}")
        print(f"Min: {img_cv.min()}, Max: {img_cv.max()}")
        print(f"Mean B: {b.mean():.2f}, G: {g.mean():.2f}, R: {r.mean():.2f}")
        print(f"Std B: {b.std():.2f}, G: {g.std():.2f}, R: {r.std():.2f}")

    unique_vals = np.unique(img_cv)
    print(f"Unique values count: {len(unique_vals)}")
    if len(unique_vals) <= 256:
        print(f"Has full 0-255 range: {len(unique_vals) == 256}")

    print(f"Is grayscale (B==G==R): {np.array_equal(b, g) and np.array_equal(g, r) if len(img_cv.shape) == 3 else 'N/A'}")
