import os
import shutil
import json
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

from volumen.segmentation.strategy import (
    HSVThresholdStrategy,
    SampledColorStrategy,
    GrabCutStrategy,
    SVDCodesStrategy,
)
from volumen.camera import extract_exif_metadata, get_focal_length
from volumen.estimator import estimate_volume

app = FastAPI(title="Plant Volume Estimator")

# Ensure the base data directory exists
VOLUME_DIR = Path.home() / "Info" / "volume"
VOLUME_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)

# Mount the static directory to serve the frontend website
app.mount("/static", StaticFiles(directory="./website/static"), name="static")

# Mount the volume directory to serve uploaded images and masks to the frontend
app.mount("/data", StaticFiles(directory=str(VOLUME_DIR)), name="data")

@app.get("/")
async def read_index():
    return FileResponse("./website/static/index.html")

def save_mask(mask: np.ndarray, path: str):
    # Convert boolean mask to 0-255 uint8 image
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img.save(path)

def _make_strategy(strategy_name: str):
    """Return the segmentation strategy object for the given name."""
    if strategy_name == "grabcut":
        return GrabCutStrategy()
    elif strategy_name == "svdcodes":
        return SVDCodesStrategy()
    else:
        return HSVThresholdStrategy()

def _save_svd_previews(
    strategy: SVDCodesStrategy,
    paths: dict,
    run_dir: str,
    resolution: int,
    alpha: int,
    view_type: str,
) -> dict:
    """
    Generate and save colour-cluster preview images for each view.
    Returns a dict mapping view → saved file path.
    """
    preview_paths = {}
    for view, img_path in paths.items():
        segmented = strategy.segment_image(
            img_path, resolution, alpha=alpha, view_type=view_type
        )
        preview_path = os.path.join(run_dir, f"svd_preview_{view}.png")
        Image.fromarray(segmented.astype(np.uint8)).save(preview_path)
        preview_paths[view] = preview_path
    return preview_paths

@app.post("/api/segment")
async def segment_images(
    photo_xy: UploadFile = File(...),
    photo_yz: UploadFile = File(...),
    photo_xz: UploadFile = File(...),
    strategy: str = Form("hsv"),
    alpha: int = Form(16),
    view_type: str = Form("segments"),
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_dir = os.path.join(VOLUME_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    paths = {
        "xy": os.path.join(run_dir, "photo_xy.jpg"),
        "yz": os.path.join(run_dir, "photo_yz.jpg"),
        "xz": os.path.join(run_dir, "photo_xz.jpg")
    }

    # Save uploaded images
    for photo, path in [(photo_xy, paths["xy"]), (photo_yz, paths["yz"]), (photo_xz, paths["xz"])]:
        with open(path, "wb") as buffer:
            shutil.copyfileobj(photo.file, buffer)
        photo.file.close()

    # Extract EXIF focal length from one of the images (assume xy)
    exif = extract_exif_metadata(paths["xy"])
    focal_length = get_focal_length(exif, default=50.0)

    resolution = 256
    strategy_obj = _make_strategy(strategy)

    extra = {}
    if strategy == "svdcodes":
        extra["alpha"] = alpha
        extra["view_type"] = view_type

    mask_xy = strategy_obj.create_mask(paths["xy"], resolution, is_contrasting_bg=True, **extra)
    mask_yz = strategy_obj.create_mask(paths["yz"], resolution, is_contrasting_bg=True, **extra)
    mask_xz = strategy_obj.create_mask(paths["xz"], resolution, is_contrasting_bg=False, **extra)

    mask_paths = {
        "xy": os.path.join(run_dir, "mask_xy.png"),
        "yz": os.path.join(run_dir, "mask_yz.png"),
        "xz": os.path.join(run_dir, "mask_xz.png")
    }

    save_mask(mask_xy, mask_paths["xy"])
    save_mask(mask_yz, mask_paths["yz"])
    save_mask(mask_xz, mask_paths["xz"])

    try:
        volume_m3 = estimate_volume(
            mask_xy=mask_xy,
            mask_yz=mask_yz,
            mask_xz=mask_xz,
            focal_length=focal_length,
            resolution=256
        )
    except Exception as e:
        print(f"Initial volume estimation failed: {e}")
        volume_m3 = 0.0

    # Build response
    response = {
        "run_id": run_id,
        "focal_length": focal_length,
        "exif": exif,
        "images": {k: f"/data/{run_id}/photo_{k}.jpg" for k in paths.keys()},
        "masks": {k: f"/data/{run_id}/mask_{k}.png" for k in mask_paths.keys()},
        "estimated_volume_m3": volume_m3,
        "svd_previews": None,
    }

    # Generate and return colour previews when SVDCodes was used
    if strategy == "svdcodes":
        ts = int(datetime.now().timestamp())
        _save_svd_previews(strategy_obj, paths, run_dir, resolution, alpha, view_type)
        response["svd_previews"] = {
            k: f"/data/{run_id}/svd_preview_{k}.png?t={ts}" for k in paths.keys()
        }

    return JSONResponse(response)

class UpdateMaskRequest(BaseModel):
    run_id: str
    view: str  # 'xy', 'yz', or 'xz'
    focal_length: float
    fg_rects: List[List[int]] = [] # list of [x, y, w, h]
    bg_rects: List[List[int]] = [] # list of [x, y, w, h]

@app.post("/api/update-mask")
async def update_mask(req: UpdateMaskRequest):
    run_dir = os.path.join(VOLUME_DIR, req.run_id)
    if not os.path.exists(run_dir):
        raise HTTPException(status_code=404, detail="Run not found")
        
    img_path = os.path.join(run_dir, f"photo_{req.view}.jpg")
    mask_path = os.path.join(run_dir, f"mask_{req.view}.png")
    
    strategy = GrabCutStrategy()
    mask = strategy.create_mask(
        img_path, 
        resolution=256, 
        fg_rects=req.fg_rects,
        bg_rects=req.bg_rects
    )
    
    save_mask(mask, mask_path)
    
    # Recalculate Volume
    masks = {}
    for v in ["xy", "yz", "xz"]:
        mp = os.path.join(run_dir, f"mask_{v}.png")
        img = Image.open(mp).convert('L')
        masks[v] = np.array(img) > 128
        
    volume_m3 = estimate_volume(
        mask_xy=masks["xy"],
        mask_yz=masks["yz"],
        mask_xz=masks["xz"],
        focal_length=req.focal_length,
        resolution=256
    )
    
    # Return cache-busting URL
    timestamp = int(datetime.now().timestamp())
    return JSONResponse({
        "mask_url": f"/data/{req.run_id}/mask_{req.view}.png?t={timestamp}",
        "estimated_volume_m3": volume_m3
    })

class ResegmentRequest(BaseModel):
    run_id: str
    strategy: str
    focal_length: float
    alpha: int = 16
    view_type: str = "segments"

@app.post("/api/resegment")
async def resegment_images(req: ResegmentRequest):
    run_dir = os.path.join(VOLUME_DIR, req.run_id)
    if not os.path.exists(run_dir):
        raise HTTPException(status_code=404, detail="Run not found")

    paths = {
        "xy": os.path.join(run_dir, "photo_xy.jpg"),
        "yz": os.path.join(run_dir, "photo_yz.jpg"),
        "xz": os.path.join(run_dir, "photo_xz.jpg")
    }

    strategy_obj = _make_strategy(req.strategy)

    extra = {}
    if req.strategy == "svdcodes":
        extra["alpha"] = req.alpha
        extra["view_type"] = req.view_type

    resolution = 256
    mask_xy = strategy_obj.create_mask(paths["xy"], resolution, is_contrasting_bg=True, **extra)
    mask_yz = strategy_obj.create_mask(paths["yz"], resolution, is_contrasting_bg=True, **extra)
    mask_xz = strategy_obj.create_mask(paths["xz"], resolution, is_contrasting_bg=False, **extra)

    mask_paths = {
        "xy": os.path.join(run_dir, "mask_xy.png"),
        "yz": os.path.join(run_dir, "mask_yz.png"),
        "xz": os.path.join(run_dir, "mask_xz.png")
    }

    save_mask(mask_xy, mask_paths["xy"])
    save_mask(mask_yz, mask_paths["yz"])
    save_mask(mask_xz, mask_paths["xz"])
    
    volume_m3 = estimate_volume(
        mask_xy=mask_xy,
        mask_yz=mask_yz,
        mask_xz=mask_xz,
        focal_length=req.focal_length,
        resolution=256
    )

    ts = int(datetime.now().timestamp())
    response = {
        "masks": {k: f"/data/{req.run_id}/mask_{k}.png?t={ts}" for k in mask_paths.keys()},
        "estimated_volume_m3": volume_m3,
        "svd_previews": None,
    }

    if req.strategy == "svdcodes":
        _save_svd_previews(strategy_obj, paths, run_dir, resolution, req.alpha, req.view_type)
        response["svd_previews"] = {
            k: f"/data/{req.run_id}/svd_preview_{k}.png?t={ts}" for k in paths.keys()
        }

    return JSONResponse(response)


# ── SVD Colour Preview ────────────────────────────────────────────────────────

class SVDPreviewRequest(BaseModel):
    run_id: str
    view: str           # 'xy', 'yz', or 'xz'
    alpha: int = 16
    view_type: str = "segments"

@app.post("/api/svd-preview")
async def svd_preview(req: SVDPreviewRequest):
    """
    Generate and return a colour-cluster preview image for one view using the
    SVDCodes algorithm.  The image is saved in the run directory so the static
    file server can serve it directly.
    """
    run_dir = os.path.join(VOLUME_DIR, req.run_id)
    if not os.path.exists(run_dir):
        raise HTTPException(status_code=404, detail="Run not found")

    img_path = os.path.join(run_dir, f"photo_{req.view}.jpg")
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail=f"Image for view '{req.view}' not found")

    strategy = SVDCodesStrategy()
    segmented = strategy.segment_image(
        img_path, resolution=256, alpha=req.alpha, view_type=req.view_type
    )

    preview_path = os.path.join(run_dir, f"svd_preview_{req.view}.png")
    Image.fromarray(segmented.astype(np.uint8)).save(preview_path)

    ts = int(datetime.now().timestamp())
    return JSONResponse({
        "preview_url": f"/data/{req.run_id}/svd_preview_{req.view}.png?t={ts}"
    })


# ── The /api/calculate-volume endpoint is kept for manual focal length updates ─

class CalculateVolumeRequest(BaseModel):
    run_id: str
    focal_length: float

@app.post("/api/calculate-volume")
async def calculate_volume(req: CalculateVolumeRequest):
    run_dir = os.path.join(VOLUME_DIR, req.run_id)
    if not os.path.exists(run_dir):
        raise HTTPException(status_code=404, detail="Run not found")

    masks = {}
    for view in ["xy", "yz", "xz"]:
        mask_path = os.path.join(run_dir, f"mask_{view}.png")
        if not os.path.exists(mask_path):
            raise HTTPException(status_code=404, detail=f"Mask for {view} not found")
        
        # Load mask and convert to boolean numpy array
        img = Image.open(mask_path).convert('L')
        # Mask was saved as 0-255 uint8
        mask_array = np.array(img) > 128
        masks[view] = mask_array

    try:
        volume_m3 = estimate_volume(
            mask_xy=masks["xy"],
            mask_yz=masks["yz"],
            mask_xz=masks["xz"],
            focal_length=req.focal_length,
            resolution=256
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Volume calculation failed: {str(e)}")

    return JSONResponse({
        "estimated_volume_m3": volume_m3
    })
