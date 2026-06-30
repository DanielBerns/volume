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

from volumen.segmentation.strategy import HSVThresholdStrategy, SampledColorStrategy
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

@app.post("/api/segment")
async def segment_images(
    photo_xy: UploadFile = File(...),
    photo_yz: UploadFile = File(...),
    photo_xz: UploadFile = File(...)
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

    # Initial segmentation with default HSV strategy
    strategy = HSVThresholdStrategy()
    resolution = 256
    
    # We assume xy and yz have contrasting bg, xz might not, as per original logic
    mask_xy = strategy.create_mask(paths["xy"], resolution, is_contrasting_bg=True)
    mask_yz = strategy.create_mask(paths["yz"], resolution, is_contrasting_bg=True)
    mask_xz = strategy.create_mask(paths["xz"], resolution, is_contrasting_bg=False)

    mask_paths = {
        "xy": os.path.join(run_dir, "mask_xy.png"),
        "yz": os.path.join(run_dir, "mask_yz.png"),
        "xz": os.path.join(run_dir, "mask_xz.png")
    }

    save_mask(mask_xy, mask_paths["xy"])
    save_mask(mask_yz, mask_paths["yz"])
    save_mask(mask_xz, mask_paths["xz"])

    return JSONResponse({
        "run_id": run_id,
        "focal_length": focal_length,
        "exif": exif,
        "images": {k: f"/data/{run_id}/photo_{k}.jpg" for k in paths.keys()},
        "masks": {k: f"/data/{run_id}/mask_{k}.png" for k in mask_paths.keys()}
    })

class UpdateMaskRequest(BaseModel):
    run_id: str
    view: str  # 'xy', 'yz', or 'xz'
    sampled_pixels: List[List[int]]  # list of [H, S, V] values
    tolerance: Optional[int] = 20

@app.post("/api/update-mask")
async def update_mask(req: UpdateMaskRequest):
    run_dir = os.path.join(VOLUME_DIR, req.run_id)
    if not os.path.exists(run_dir):
        raise HTTPException(status_code=404, detail="Run not found")
        
    img_path = os.path.join(run_dir, f"photo_{req.view}.jpg")
    mask_path = os.path.join(run_dir, f"mask_{req.view}.png")
    
    strategy = SampledColorStrategy()
    mask = strategy.create_mask(
        img_path, 
        resolution=256, 
        sampled_pixels=req.sampled_pixels, 
        tolerance=req.tolerance
    )
    
    save_mask(mask, mask_path)
    
    # Return cache-busting URL
    timestamp = int(datetime.now().timestamp())
    return JSONResponse({
        "mask_url": f"/data/{req.run_id}/mask_{req.view}.png?t={timestamp}"
    })

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
