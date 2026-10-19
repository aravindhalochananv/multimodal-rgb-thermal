"""
FastAPI Backend Server for Multimodal RGB-Thermal Human Detection.
Provides REST APIs for model inference, comparison, sample retrieval, and metrics.
"""

import base64
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO

repo_root = Path(__file__).resolve().parents[1]

from src.config import (
    DATA_VISIBLE_TEST,
    DATA_INFRARED_TEST,
    OUTPUT_RGB,
    OUTPUT_THERMAL,
    OUTPUT_MULTIMODAL,
    OUTPUT_COMPARISON,
)
from src.models.fusion_model import PixelFusionYOLO

app = FastAPI(
    title="Multimodal RGB-Thermal Detection API",
    description="Backend API for YOLOv8 RGB, Thermal, and Multimodal Pedestrian Detection",
    version="1.0.0",
)

# Enable CORS for React frontend (Vite runs on 5173 by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache to avoid reloading on every request
MODELS_CACHE: Dict[str, Any] = {}


def get_cached_model(model_key: str):
    """Load and cache models in memory."""
    if model_key in MODELS_CACHE:
        return MODELS_CACHE[model_key]

    weights_map = {
        "rgb": OUTPUT_RGB / "yolov8n" / "train" / "weights" / "best.pt",
        "thermal": OUTPUT_THERMAL / "yolov8n" / "train" / "weights" / "best.pt",
        "multimodal": OUTPUT_MULTIMODAL / "yolov8n" / "train" / "weights" / "best.pt",
    }

    # Fallbacks to colab/processed_llvip_verified if needed
    colab_fallback = {
        "rgb": repo_root / "colab" / "processed_llvip_verified" / "rgb_best.pt",
        "thermal": repo_root / "colab" / "processed_llvip_verified" / "thermal_best.pt",
        "multimodal": repo_root / "colab" / "processed_llvip_verified" / "fusion_best.pt",
    }

    path = weights_map.get(model_key)
    if path is None or not path.exists():
        path = colab_fallback.get(model_key)

    if path is None or not path.exists():
        path = repo_root / "yolov8n.pt"

    if model_key == "multimodal":
        m = PixelFusionYOLO(weights_path=path)
    else:
        m = YOLO(str(path))

    MODELS_CACHE[model_key] = m
    return m


def image_to_base64(img_bgr: np.ndarray) -> str:
    """Encode OpenCV BGR image to base64 JPEG data URL."""
    _, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def parse_upload_image(file_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into OpenCV BGR numpy array."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return img


@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "timestamp": time.time(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "models_available": ["rgb", "thermal", "multimodal"],
    }


@app.get("/api/models")
def get_models_info():
    """Return model descriptions, architecture, and Colab training details."""
    return [
        {
            "id": "rgb",
            "name": "RGB-Only Baseline",
            "description": "YOLOv8n trained on visible spectrum images. Captures color and fine texture but degrades under low-light.",
            "epochs": 10,
            "mAP50": 0.8596,
            "mAP50_95": 0.4692,
            "precision": 0.8904,
            "recall": 0.7870,
            "latency_ms": 54.07,
            "weights": "outputs/rgb/yolov8n/train/weights/best.pt",
        },
        {
            "id": "thermal",
            "name": "Thermal-Only Baseline",
            "description": "YOLOv8n trained on infrared images. Captures human body heat emissions regardless of darkness or glare.",
            "epochs": 10,
            "mAP50": 0.9222,
            "mAP50_95": 0.5658,
            "precision": 0.9267,
            "recall": 0.8418,
            "latency_ms": 52.15,
            "weights": "outputs/thermal/yolov8n/train/weights/best.pt",
        },
        {
            "id": "multimodal",
            "name": "50:50 Multimodal Fusion",
            "description": "YOLOv8n trained on blended RGB+Thermal representations. Combines visible context with infrared heat signatures.",
            "epochs": 10,
            "mAP50": 0.9150,
            "mAP50_95": 0.5505,
            "precision": 0.9218,
            "recall": 0.8295,
            "latency_ms": 38.19,
            "weights": "outputs/multimodal/yolov8n/train/weights/best.pt",
        },
    ]


@app.get("/api/samples")
def get_sample_list():
    """Return a curated list of paired test samples from LLVIP."""
    samples = []
    if DATA_VISIBLE_TEST.exists():
        rgb_files = sorted(list(DATA_VISIBLE_TEST.glob("*.jpg")))[:20]
        for f in rgb_files:
            samples.append({
                "id": f.stem,
                "filename": f.name,
                "rgb_url": f"/api/sample-image/rgb/{f.name}",
                "thermal_url": f"/api/sample-image/thermal/{f.name}",
            })
    return samples


@app.get("/api/sample-image/{modality}/{filename}")
def get_sample_image(modality: str, filename: str):
    """Serve sample image from dataset."""
    if modality == "rgb":
        p = DATA_VISIBLE_TEST / filename
    else:
        p = DATA_INFRARED_TEST / filename

    if not p.exists():
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(str(p), media_type="image/jpeg")


@app.post("/api/predict")
async def predict_endpoint(
    modality: str = Form("rgb"),
    conf: float = Form(0.25),
    iou: float = Form(0.50),
    alpha: float = Form(0.50),
    sample_id: Optional[str] = Form(None),
    rgb_file: Optional[UploadFile] = File(None),
    thermal_file: Optional[UploadFile] = File(None),
):
    """Run single or multimodal prediction."""
    t0 = time.perf_counter()

    # Load images
    rgb_img = None
    th_img = None

    if sample_id:
        rgb_path = DATA_VISIBLE_TEST / f"{sample_id}.jpg"
        th_path = DATA_INFRARED_TEST / f"{sample_id}.jpg"
        if rgb_path.exists():
            rgb_img = cv2.imread(str(rgb_path))
        if th_path.exists():
            th_img = cv2.imread(str(th_path))

    if rgb_file is not None:
        rgb_bytes = await rgb_file.read()
        rgb_img = parse_upload_image(rgb_bytes)

    if thermal_file is not None:
        th_bytes = await thermal_file.read()
        th_img = parse_upload_image(th_bytes)

    # Process based on modality
    if modality == "rgb":
        if rgb_img is None:
            raise HTTPException(status_code=400, detail="RGB image is required for rgb modality")
        model = get_cached_model("rgb")
        res = model.predict(rgb_img, imgsz=320, conf=conf, iou=iou, verbose=False)[0]
        vis_img = res.plot()
        boxes_data = res.boxes

    elif modality == "thermal":
        if th_img is None:
            if rgb_img is not None:
                th_img = rgb_img
            else:
                raise HTTPException(status_code=400, detail="Thermal image is required for thermal modality")
        model = get_cached_model("thermal")
        res = model.predict(th_img, imgsz=320, conf=conf, iou=iou, verbose=False)[0]
        vis_img = res.plot()
        boxes_data = res.boxes

    else:
        # Multimodal fusion
        if rgb_img is None or th_img is None:
            raise HTTPException(status_code=400, detail="Both RGB and Thermal images are required for multimodal fusion")
        fusion_model = get_cached_model("multimodal")
        fusion_model.alpha = alpha
        res = fusion_model.predict(rgb_img, th_img, imgsz=320, conf=conf, iou=iou, verbose=False)[0]
        vis_img = res.plot()
        boxes_data = res.boxes

    latency_ms = (time.perf_counter() - t0) * 1000

    # Format detections
    detections = []
    if len(boxes_data) > 0:
        xyxy = boxes_data.xyxy.cpu().numpy()
        confs = boxes_data.conf.cpu().numpy()
        for i in range(len(confs)):
            detections.append({
                "id": i + 1,
                "bbox": [round(float(v), 1) for v in xyxy[i]],
                "confidence": round(float(confs[i]), 3),
                "class": "person",
            })

    return {
        "modality": modality,
        "count": len(detections),
        "latency_ms": round(latency_ms, 2),
        "fps": round(1000.0 / latency_ms, 1) if latency_ms > 0 else 0,
        "detections": detections,
        "image_base64": image_to_base64(vis_img),
    }


@app.post("/api/compare")
async def compare_endpoint(
    conf: float = Form(0.25),
    iou: float = Form(0.50),
    sample_id: Optional[str] = Form(None),
    rgb_file: Optional[UploadFile] = File(None),
    thermal_file: Optional[UploadFile] = File(None),
):
    """Run all 3 models (RGB, Thermal, Fusion) and return comparative detections."""
    t0 = time.perf_counter()

    rgb_img = None
    th_img = None

    if sample_id:
        rgb_path = DATA_VISIBLE_TEST / f"{sample_id}.jpg"
        th_path = DATA_INFRARED_TEST / f"{sample_id}.jpg"
        if rgb_path.exists():
            rgb_img = cv2.imread(str(rgb_path))
        if th_path.exists():
            th_img = cv2.imread(str(th_path))

    if rgb_file is not None:
        rgb_img = parse_upload_image(await rgb_file.read())
    if thermal_file is not None:
        th_img = parse_upload_image(await thermal_file.read())

    if rgb_img is None or th_img is None:
        raise HTTPException(status_code=400, detail="Both RGB and Thermal images are required for comparison")

    # 1. RGB Model
    t_rgb0 = time.perf_counter()
    m_rgb = get_cached_model("rgb")
    res_rgb = m_rgb.predict(rgb_img, imgsz=320, conf=conf, iou=iou, verbose=False)[0]
    lat_rgb = (time.perf_counter() - t_rgb0) * 1000

    # 2. Thermal Model
    t_th0 = time.perf_counter()
    m_th = get_cached_model("thermal")
    res_th = m_th.predict(th_img, imgsz=320, conf=conf, iou=iou, verbose=False)[0]
    lat_th = (time.perf_counter() - t_th0) * 1000

    # 3. Fusion Model (50:50)
    t_fu0 = time.perf_counter()
    m_fu = get_cached_model("multimodal")
    m_fu.alpha = 0.5
    res_fu = m_fu.predict(rgb_img, th_img, imgsz=320, conf=conf, iou=iou, verbose=False)[0]
    lat_fu = (time.perf_counter() - t_fu0) * 1000

    total_latency = (time.perf_counter() - t0) * 1000

    def extract_boxes(boxes_obj):
        dets = []
        if len(boxes_obj) > 0:
            xyxy = boxes_obj.xyxy.cpu().numpy()
            confs = boxes_obj.conf.cpu().numpy()
            for i in range(len(confs)):
                dets.append({
                    "id": i + 1,
                    "bbox": [round(float(v), 1) for v in xyxy[i]],
                    "confidence": round(float(confs[i]), 3),
                    "class": "person",
                })
        return dets

    return {
        "total_latency_ms": round(total_latency, 2),
        "rgb": {
            "count": len(res_rgb.boxes),
            "latency_ms": round(lat_rgb, 2),
            "detections": extract_boxes(res_rgb.boxes),
            "image_base64": image_to_base64(res_rgb.plot()),
        },
        "thermal": {
            "count": len(res_th.boxes),
            "latency_ms": round(lat_th, 2),
            "detections": extract_boxes(res_th.boxes),
            "image_base64": image_to_base64(res_th.plot()),
        },
        "multimodal": {
            "count": len(res_fu.boxes),
            "latency_ms": round(lat_fu, 2),
            "detections": extract_boxes(res_fu.boxes),
            "image_base64": image_to_base64(res_fu.plot()),
        },
    }


@app.get("/api/metrics")
def get_metrics():
    """Return complete benchmark metrics comparison."""
    cmp_path = OUTPUT_COMPARISON / "comparison.json"
    if cmp_path.exists():
        with open(cmp_path) as f:
            data = json.load(f)
        return data

    # Hardcoded verified fallback from Colab test comparison
    return [
        {
            "modality": "rgb",
            "model": "yolov8n",
            "precision": 0.8904,
            "recall": 0.7870,
            "f1_score": 0.8356,
            "mAP50": 0.8596,
            "mAP50_95": 0.4692,
            "inference_time": {"avg_per_image_ms": 54.07},
        },
        {
            "modality": "thermal",
            "model": "yolov8n",
            "precision": 0.9267,
            "recall": 0.8418,
            "f1_score": 0.8822,
            "mAP50": 0.9222,
            "mAP50_95": 0.5658,
            "inference_time": {"avg_per_image_ms": 52.15},
        },
        {
            "modality": "multimodal",
            "model": "yolov8n",
            "precision": 0.9218,
            "recall": 0.8295,
            "f1_score": 0.8732,
            "mAP50": 0.9150,
            "mAP50_95": 0.5505,
            "inference_time": {"avg_per_image_ms": 38.19},
        },
    ]


@app.get("/api/static-plot/{plot_name}")
def get_static_plot(plot_name: str):
    """Serve comparison plots."""
    allowed = {
        "metrics_comparison_bar.png": OUTPUT_COMPARISON / "metrics_comparison_bar.png",
        "map_comparison.png": OUTPUT_COMPARISON / "map_comparison.png",
        "training_curves_comparison.png": OUTPUT_COMPARISON / "training_curves_comparison.png",
        "rgb_results.png": OUTPUT_RGB / "yolov8n" / "train" / "results.png",
        "thermal_results.png": OUTPUT_THERMAL / "yolov8n" / "train" / "results.png",
        "fusion_results.png": OUTPUT_MULTIMODAL / "yolov8n" / "train" / "results.png",
    }
    p = allowed.get(plot_name)
    if p and p.exists():
        return FileResponse(str(p), media_type="image/png")
    raise HTTPException(status_code=404, detail="Plot not found")


@app.get("/api/qualitative-panels")
def get_qualitative_panels():
    """List qualitative comparison panels from outputs/comparison/predictions."""
    p_dir = OUTPUT_COMPARISON / "predictions"
    panels = []
    if p_dir.exists():
        for f in sorted(list(p_dir.glob("*_comparison.jpg"))):
            panels.append({
                "id": f.stem,
                "filename": f.name,
                "url": f"/api/qualitative-panel-file/{f.name}",
            })
    return panels


@app.get("/api/qualitative-panel-file/{filename}")
def get_panel_file(filename: str):
    p = OUTPUT_COMPARISON / "predictions" / filename
    if p.exists():
        return FileResponse(str(p), media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Panel not found")


# Mount React static build if available
from fastapi.staticfiles import StaticFiles
dist_path = repo_root / "frontend" / "dist"
if dist_path.exists():
    app.mount("/", StaticFiles(directory=str(dist_path), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
