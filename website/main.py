import os
import shutil
import json
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict

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

RESOLUTION = 256

@app.get("/")
async def read_index():
    return FileResponse("./website/static/index.html")


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_mask(mask: np.ndarray, path: str):
    """Save a boolean mask as a grayscale PNG (0/255)."""
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


def _run_svd_clusters(
    image_path: str,
    run_id: str,
    run_dir: str,
    view: str,
    resolution: int,
    alpha: int,
    min_cluster_pct: float,
) -> list:
    """
    Run SVDCodes clustering on one view image.

    Writes to disk:
      • svd_codes_{view}.npy          — flat codes array for mask reconstruction
      • svd_cluster_{view}_{code}.png — per-cluster thumbnail (orig pixels or black)

    Returns
    -------
    list[dict]  Cluster metadata for clusters that pass the size threshold.
    Each entry: { code, pixel_count, pct_total, rep_color, thumbnail_url }
    (numpy arrays are not included — they are only used locally here).
    """
    strategy = SVDCodesStrategy()
    cluster_data, codes = strategy.get_cluster_masks(image_path, resolution, alpha)

    # Persist the codes array so compute-from-clusters can reconstruct masks
    np.save(os.path.join(run_dir, f"svd_codes_{view}.npy"), codes)

    total_pixels = resolution * resolution
    min_pixels = int(total_pixels * min_cluster_pct / 100.0)

    meta = []
    for item in cluster_data:
        if item["pixel_count"] < min_pixels:
            continue  # skip tiny clusters (noise)

        thumb_path = os.path.join(run_dir, f"svd_cluster_{view}_{item['code']}.png")
        Image.fromarray(item["thumbnail"].astype(np.uint8)).save(thumb_path)

        pct = round(item["pixel_count"] / total_pixels * 100, 2)
        meta.append({
            "code":          item["code"],
            "pixel_count":   item["pixel_count"],
            "pct_total":     pct,
            "rep_color":     item["rep_color"],
            "thumbnail_url": f"/data/{run_id}/svd_cluster_{view}_{item['code']}.png",
        })

    return meta


def _initial_svd_mask(
    run_dir: str,
    view: str,
    resolution: int,
    visible_codes: list,
) -> np.ndarray:
    """
    Build the initial combined boolean mask from the visible (above-threshold)
    SVD cluster codes, using the saved codes array.
    """
    codes = np.load(os.path.join(run_dir, f"svd_codes_{view}.npy"))
    if not visible_codes:
        return np.zeros((resolution, resolution), dtype=bool)
    combined = np.zeros(resolution * resolution, dtype=bool)
    for code in visible_codes:
        combined |= codes == code
    return combined.reshape(resolution, resolution)


# ── Segment (upload) ──────────────────────────────────────────────────────────

@app.post("/api/segment")
async def segment_images(
    photo_xy: UploadFile = File(...),
    photo_yz: UploadFile = File(...),
    photo_xz: UploadFile = File(...),
    strategy: str = Form("hsv"),
    alpha: int = Form(16),
    min_cluster_pct: float = Form(1.0),
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_dir = os.path.join(VOLUME_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    photo_paths = {
        "xy": os.path.join(run_dir, "photo_xy.jpg"),
        "yz": os.path.join(run_dir, "photo_yz.jpg"),
        "xz": os.path.join(run_dir, "photo_xz.jpg"),
    }

    for photo, path in [
        (photo_xy, photo_paths["xy"]),
        (photo_yz, photo_paths["yz"]),
        (photo_xz, photo_paths["xz"]),
    ]:
        with open(path, "wb") as buf:
            shutil.copyfileobj(photo.file, buf)
        photo.file.close()

    exif = extract_exif_metadata(photo_paths["xy"])
    focal_length = get_focal_length(exif, default=50.0)

    strategy_obj = _make_strategy(strategy)
    extra = {}
    if strategy == "svdcodes":
        extra["alpha"] = alpha

    masks: dict[str, np.ndarray] = {}
    clusters: dict[str, list] = {}

    for view, img_path in photo_paths.items():
        if strategy == "svdcodes":
            view_clusters = _run_svd_clusters(
                img_path, run_id, run_dir, view, RESOLUTION, alpha, min_cluster_pct
            )
            clusters[view] = view_clusters
            visible_codes = [c["code"] for c in view_clusters]
            masks[view] = _initial_svd_mask(run_dir, view, RESOLUTION, visible_codes)
        else:
            is_contrast = view != "xz"
            masks[view] = strategy_obj.create_mask(
                img_path, RESOLUTION, is_contrasting_bg=is_contrast, **extra
            )

    if strategy == "svdcodes":
        # Persist cluster metadata so /api/svd-clusters can reload it cheaply
        clusters_path = os.path.join(run_dir, "svd_clusters.json")
        with open(clusters_path, "w") as f:
            json.dump({"alpha": alpha, "min_cluster_pct": min_cluster_pct, "clusters": clusters}, f)

    mask_paths = {
        v: os.path.join(run_dir, f"mask_{v}.png") for v in ["xy", "yz", "xz"]
    }
    for v, mask in masks.items():
        save_mask(mask, mask_paths[v])

    try:
        volume_m3 = estimate_volume(
            mask_xy=masks["xy"],
            mask_yz=masks["yz"],
            mask_xz=masks["xz"],
            focal_length=focal_length,
            resolution=RESOLUTION,
        )
    except Exception as e:
        print(f"Initial volume estimation failed: {e}")
        volume_m3 = 0.0

    ts = int(datetime.now().timestamp())
    return JSONResponse({
        "run_id":              run_id,
        "focal_length":        focal_length,
        "exif":                exif,
        "images":              {v: f"/data/{run_id}/photo_{v}.jpg" for v in photo_paths},
        "masks":               {v: f"/data/{run_id}/mask_{v}.png?t={ts}" for v in mask_paths},
        "estimated_volume_m3": volume_m3,
        "clusters":            clusters if strategy == "svdcodes" else None,
    })


# ── Update mask (GrabCut interactive refinement) ──────────────────────────────

class UpdateMaskRequest(BaseModel):
    run_id: str
    view: str  # 'xy', 'yz', or 'xz'
    focal_length: float
    fg_rects: List[List[int]] = []
    bg_rects: List[List[int]] = []

@app.post("/api/update-mask")
async def update_mask(req: UpdateMaskRequest):
    run_dir = os.path.join(VOLUME_DIR, req.run_id)
    if not os.path.exists(run_dir):
        raise HTTPException(status_code=404, detail="Run not found")

    img_path  = os.path.join(run_dir, f"photo_{req.view}.jpg")
    mask_path = os.path.join(run_dir, f"mask_{req.view}.png")

    strategy = GrabCutStrategy()
    mask = strategy.create_mask(
        img_path,
        resolution=RESOLUTION,
        fg_rects=req.fg_rects,
        bg_rects=req.bg_rects,
    )
    save_mask(mask, mask_path)

    masks = {}
    for v in ["xy", "yz", "xz"]:
        img = Image.open(os.path.join(run_dir, f"mask_{v}.png")).convert("L")
        masks[v] = np.array(img) > 128

    volume_m3 = estimate_volume(
        mask_xy=masks["xy"],
        mask_yz=masks["yz"],
        mask_xz=masks["xz"],
        focal_length=req.focal_length,
        resolution=RESOLUTION,
    )

    ts = int(datetime.now().timestamp())
    return JSONResponse({
        "mask_url":            f"/data/{req.run_id}/mask_{req.view}.png?t={ts}",
        "estimated_volume_m3": volume_m3,
    })


# ── Resegment (re-run algorithm on existing run) ──────────────────────────────

class ResegmentRequest(BaseModel):
    run_id: str
    strategy: str
    focal_length: float
    alpha: int = 16
    min_cluster_pct: float = 1.0

@app.post("/api/resegment")
async def resegment_images(req: ResegmentRequest):
    run_dir = os.path.join(VOLUME_DIR, req.run_id)
    if not os.path.exists(run_dir):
        raise HTTPException(status_code=404, detail="Run not found")

    photo_paths = {
        v: os.path.join(run_dir, f"photo_{v}.jpg") for v in ["xy", "yz", "xz"]
    }

    strategy_obj = _make_strategy(req.strategy)
    extra = {}
    if req.strategy == "svdcodes":
        extra["alpha"] = req.alpha

    masks: dict[str, np.ndarray] = {}
    clusters: dict[str, list] = {}

    for view, img_path in photo_paths.items():
        if req.strategy == "svdcodes":
            view_clusters = _run_svd_clusters(
                img_path, req.run_id, run_dir, view,
                RESOLUTION, req.alpha, req.min_cluster_pct,
            )
            clusters[view] = view_clusters
            visible_codes = [c["code"] for c in view_clusters]
            masks[view] = _initial_svd_mask(run_dir, view, RESOLUTION, visible_codes)
        else:
            is_contrast = view != "xz"
            masks[view] = strategy_obj.create_mask(
                img_path, RESOLUTION, is_contrasting_bg=is_contrast, **extra
            )

    if req.strategy == "svdcodes":
        clusters_path = os.path.join(run_dir, "svd_clusters.json")
        with open(clusters_path, "w") as f:
            json.dump({
                "alpha": req.alpha,
                "min_cluster_pct": req.min_cluster_pct,
                "clusters": clusters,
            }, f)

    for v, mask in masks.items():
        save_mask(mask, os.path.join(run_dir, f"mask_{v}.png"))

    volume_m3 = estimate_volume(
        mask_xy=masks["xy"],
        mask_yz=masks["yz"],
        mask_xz=masks["xz"],
        focal_length=req.focal_length,
        resolution=RESOLUTION,
    )

    ts = int(datetime.now().timestamp())
    return JSONResponse({
        "masks":               {v: f"/data/{req.run_id}/mask_{v}.png?t={ts}" for v in masks},
        "estimated_volume_m3": volume_m3,
        "clusters":            clusters if req.strategy == "svdcodes" else None,
    })


# ── SVD Clusters (on-demand refresh) ─────────────────────────────────────────

class SVDClustersRequest(BaseModel):
    run_id: str
    alpha: int = 16
    min_cluster_pct: float = 1.0

@app.post("/api/svd-clusters")
async def svd_clusters(req: SVDClustersRequest):
    """
    Re-run SVDCodes clustering on an existing run's images without touching
    the saved masks or volume estimate.  Useful when the user changes alpha
    or min_cluster_pct without wanting a full resegment.
    """
    run_dir = os.path.join(VOLUME_DIR, req.run_id)
    if not os.path.exists(run_dir):
        raise HTTPException(status_code=404, detail="Run not found")

    clusters: dict[str, list] = {}
    for view in ["xy", "yz", "xz"]:
        img_path = os.path.join(run_dir, f"photo_{view}.jpg")
        if not os.path.exists(img_path):
            raise HTTPException(status_code=404, detail=f"Photo for view '{view}' not found")
        clusters[view] = _run_svd_clusters(
            img_path, req.run_id, run_dir, view,
            RESOLUTION, req.alpha, req.min_cluster_pct,
        )

    clusters_path = os.path.join(run_dir, "svd_clusters.json")
    with open(clusters_path, "w") as f:
        json.dump({
            "alpha": req.alpha,
            "min_cluster_pct": req.min_cluster_pct,
            "clusters": clusters,
        }, f)

    return JSONResponse({"clusters": clusters})


# ── Compute Volume from Cluster Selection ─────────────────────────────────────

class ComputeFromClustersRequest(BaseModel):
    run_id: str
    focal_length: float
    selected_codes: Dict[str, List[int]]  # {"xy": [42, 17], "yz": [42], "xz": [99]}

@app.post("/api/compute-from-clusters")
async def compute_from_clusters(req: ComputeFromClustersRequest):
    """
    Merge the user-selected SVD cluster masks for each view, save the combined
    masks, and return the updated volume estimate.
    """
    run_dir = os.path.join(VOLUME_DIR, req.run_id)
    if not os.path.exists(run_dir):
        raise HTTPException(status_code=404, detail="Run not found")

    masks: dict[str, np.ndarray] = {}
    for view in ["xy", "yz", "xz"]:
        codes_path = os.path.join(run_dir, f"svd_codes_{view}.npy")
        if not os.path.exists(codes_path):
            raise HTTPException(
                status_code=400,
                detail=f"SVD codes for view '{view}' not found. "
                       "Run the SVDCodes algorithm first.",
            )

        codes = np.load(codes_path)  # flat uint8 (resolution²,)
        selected = req.selected_codes.get(view, [])

        if not selected:
            combined = np.zeros(RESOLUTION * RESOLUTION, dtype=bool)
        else:
            combined = np.zeros(RESOLUTION * RESOLUTION, dtype=bool)
            for code in selected:
                combined |= codes == code

        masks[view] = combined.reshape(RESOLUTION, RESOLUTION)
        save_mask(masks[view], os.path.join(run_dir, f"mask_{view}.png"))

    try:
        volume_m3 = estimate_volume(
            mask_xy=masks["xy"],
            mask_yz=masks["yz"],
            mask_xz=masks["xz"],
            focal_length=req.focal_length,
            resolution=RESOLUTION,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Volume calculation failed: {str(e)}")

    ts = int(datetime.now().timestamp())
    return JSONResponse({
        "masks":               {v: f"/data/{req.run_id}/mask_{v}.png?t={ts}" for v in masks},
        "estimated_volume_m3": volume_m3,
    })


# ── Calculate Volume (focal-length update only) ───────────────────────────────

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
        img = Image.open(mask_path).convert("L")
        masks[view] = np.array(img) > 128

    try:
        volume_m3 = estimate_volume(
            mask_xy=masks["xy"],
            mask_yz=masks["yz"],
            mask_xz=masks["xz"],
            focal_length=req.focal_length,
            resolution=RESOLUTION,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Volume calculation failed: {str(e)}")

    return JSONResponse({"estimated_volume_m3": volume_m3})
