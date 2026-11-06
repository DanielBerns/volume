import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from volumen.camera import extract_exif_metadata, get_focal_length
from volumen.estimator import estimate_volume, estimate_volume_with_trihedron
from volumen.segmentation.strategy import (
    GrabCutStrategy,
    HSVThresholdStrategy,
    SVDCodesStrategy,
    TrihedronPlantSegmentationStrategy,
)
from volumen.trihedron import (
    TrihedronKeypoints,
    draw_trihedron_overlay,
    find_trihedron_keypoints,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Plant Volume Estimator")

# Base directories using pathlib exclusively
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "website" / "static"
VOLUME_DIR = Path.home() / "Info" / "volume"
VOLUME_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/data", StaticFiles(directory=str(VOLUME_DIR)), name="data")

RESOLUTION = 256


@app.get("/")
async def read_index():
    return FileResponse(STATIC_DIR / "index.html")


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_mask(mask: np.ndarray, path: Path) -> None:
    """Save a boolean mask as a grayscale PNG (0/255)."""
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img.save(path)


def _make_strategy(strategy_name: str):
    """Return the segmentation strategy object for the given name."""
    if strategy_name == "trihedron":
        return TrihedronPlantSegmentationStrategy()
    elif strategy_name == "grabcut":
        return GrabCutStrategy()
    elif strategy_name == "svdcodes":
        return SVDCodesStrategy()
    else:
        return HSVThresholdStrategy()


def _run_svd_clusters(
    image_path: Path,
    run_id: str,
    run_dir: Path,
    view: str,
    resolution: int,
    alpha: int,
    min_cluster_pct: float,
) -> list:
    strategy = SVDCodesStrategy()
    cluster_data, codes = strategy.get_cluster_masks(str(image_path), resolution, alpha)

    np.save(run_dir / f"svd_codes_{view}.npy", codes)

    total_pixels = resolution * resolution
    min_pixels = int(total_pixels * min_cluster_pct / 100.0)

    meta = []
    for item in cluster_data:
        if item["pixel_count"] < min_pixels:
            continue

        thumb_path = run_dir / f"svd_cluster_{view}_{item['code']}.png"
        Image.fromarray(item["thumbnail"].astype(np.uint8)).save(thumb_path)

        pct = round(item["pixel_count"] / total_pixels * 100, 2)
        meta.append({
            "code": item["code"],
            "pixel_count": item["pixel_count"],
            "pct_total": pct,
            "rep_color": item["rep_color"],
            "thumbnail_url": f"/data/{run_id}/svd_cluster_{view}_{item['code']}.png",
        })

    return meta


def _initial_svd_mask(
    run_dir: Path,
    view: str,
    resolution: int,
    visible_codes: list,
) -> np.ndarray:
    codes = np.load(run_dir / f"svd_codes_{view}.npy")
    if not visible_codes:
        return np.zeros((resolution, resolution), dtype=bool)
    combined = np.zeros(resolution * resolution, dtype=bool)
    for code in visible_codes:
        combined |= codes == code
    return combined.reshape(resolution, resolution)


def _compute_volumes(
    run_dir: Path,
    masks: Dict[str, np.ndarray],
    focal_length: float,
    arm_length_cm: float = 20.0,
) -> tuple[float, float, Dict[str, Any]]:
    """Compute (volume_cm3, volume_m3, recon_meta).

    Prioritizes calibrated trihedron reconstruction if keypoints exist for all 3 views.
    Falls back to legacy visual hull with focal length.
    """
    tri_kps: Dict[str, Optional[TrihedronKeypoints]] = {}
    for v in ["xy", "yz", "xz"]:
        photo_file = run_dir / f"photo_{v}.jpg"
        if photo_file.exists():
            img_bgr = cv2.imread(str(photo_file))
            if img_bgr is not None:
                tri_kps[v] = find_trihedron_keypoints(img_bgr, view=v, arm_length_cm=arm_length_cm)

    recon_meta: Dict[str, Any] = {}
    volume_cm3 = 0.0
    volume_m3 = 0.0

    if all(tri_kps.get(v) is not None for v in ["xy", "yz", "xz"]):
        try:
            volume_cm3, recon_meta = estimate_volume_with_trihedron(
                mask_xy=masks["xy"],
                mask_yz=masks["yz"],
                mask_xz=masks["xz"],
                kp_xy=tri_kps["xy"],
                kp_yz=tri_kps["yz"],
                kp_xz=tri_kps["xz"],
                arm_length_cm=arm_length_cm,
            )
            volume_m3 = volume_cm3 / 1e6
            return volume_cm3, volume_m3, recon_meta
        except Exception:
            logger.exception("Trihedron volume reconstruction failed; falling back to legacy projection")

    try:
        volume_m3 = estimate_volume(
            mask_xy=masks["xy"],
            mask_yz=masks["yz"],
            mask_xz=masks["xz"],
            focal_length=focal_length,
            resolution=RESOLUTION,
        )
        volume_cm3 = volume_m3 * 1e6
    except Exception:
        logger.exception("Legacy volume estimation failed")
        volume_m3 = 0.0
        volume_cm3 = 0.0

    return volume_cm3, volume_m3, recon_meta


# ── Segment (upload) ──────────────────────────────────────────────────────────

@app.post("/api/segment")
async def segment_images(
    photo_xy: UploadFile = File(...),
    photo_yz: UploadFile = File(...),
    photo_xz: UploadFile = File(...),
    strategy: str = Form("trihedron"),
    arm_length_cm: float = Form(20.0),
    alpha: int = Form(16),
    min_cluster_pct: float = Form(1.0),
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_dir = VOLUME_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    photo_paths = {
        "xy": run_dir / "photo_xy.jpg",
        "yz": run_dir / "photo_yz.jpg",
        "xz": run_dir / "photo_xz.jpg",
    }
    for photo, path in [
        (photo_xy, photo_paths["xy"]),
        (photo_yz, photo_paths["yz"]),
        (photo_xz, photo_paths["xz"]),
    ]:
        with open(path, "wb") as buf:
            shutil.copyfileobj(photo.file, buf)
        photo.file.close()

    exif = extract_exif_metadata(str(photo_paths["xy"]))
    focal_length = get_focal_length(exif, default=50.0)

    strategy_obj = _make_strategy(strategy)
    masks: Dict[str, np.ndarray] = {}
    clusters: Dict[str, list] = {}
    trihedron_data: Dict[str, Any] = {}
    overlays: Dict[str, str] = {}

    tri_kps: Dict[str, Optional[TrihedronKeypoints]] = {}

    for view, img_path in photo_paths.items():
        img_bgr = cv2.imread(str(img_path))
        kp = find_trihedron_keypoints(
            img_bgr,
            view=view,
            arm_length_cm=arm_length_cm,
            focal_length_mm=focal_length,
        )
        tri_kps[view] = kp
        if kp is not None:
            overlay_bgr = draw_trihedron_overlay(img_bgr, kp)
            overlay_file = run_dir / f"overlay_{view}.jpg"
            cv2.imwrite(str(overlay_file), overlay_bgr)
            overlays[view] = f"/data/{run_id}/overlay_{view}.jpg"
            trihedron_data[view] = {
                "vertex": kp.vertex,
                "arm_length_px": kp.arm_length_px,
                "scale_px_per_cm": kp.px_per_cm,
                "reproj_error_px": kp.reproj_error,
            }

    # Generate masks
    for view, img_path in photo_paths.items():
        if strategy == "svdcodes":
            view_clusters = _run_svd_clusters(
                img_path, run_id, run_dir, view, RESOLUTION, alpha, min_cluster_pct
            )
            clusters[view] = view_clusters
            visible_codes = [c["code"] for c in view_clusters]
            masks[view] = _initial_svd_mask(run_dir, view, RESOLUTION, visible_codes)
        elif strategy == "trihedron":
            masks[view] = strategy_obj.create_mask(str(img_path), RESOLUTION)
        else:
            is_contrast = view != "xz"
            masks[view] = strategy_obj.create_mask(
                str(img_path), RESOLUTION, is_contrasting_bg=is_contrast
            )

    if strategy == "svdcodes":
        clusters_path = run_dir / "svd_clusters.json"
        with open(clusters_path, "w") as f:
            json.dump({"alpha": alpha, "min_cluster_pct": min_cluster_pct, "clusters": clusters}, f)

    mask_paths = {
        v: run_dir / f"mask_{v}.png" for v in ["xy", "yz", "xz"]
    }
    for v, mask in masks.items():
        save_mask(mask, mask_paths[v])

    volume_cm3, volume_m3, recon_meta = _compute_volumes(
        run_dir=run_dir,
        masks=masks,
        focal_length=focal_length,
        arm_length_cm=arm_length_cm,
    )

    # Save trihedron metadata to run_dir
    tri_meta_file = run_dir / "trihedron_metadata.json"
    with open(tri_meta_file, "w") as f:
        json.dump({
            "arm_length_cm": arm_length_cm,
            "trihedron_data": trihedron_data,
            "recon_meta": recon_meta,
            "volume_cm3": volume_cm3,
            "volume_m3": volume_m3,
        }, f, indent=2)

    ts = int(datetime.now().timestamp())
    return JSONResponse({
        "run_id": run_id,
        "focal_length": focal_length,
        "arm_length_cm": arm_length_cm,
        "exif": exif,
        "images": {v: f"/data/{run_id}/photo_{v}.jpg" for v in photo_paths},
        "masks": {v: f"/data/{run_id}/mask_{v}.png?t={ts}" for v in mask_paths},
        "overlays": overlays if overlays else None,
        "trihedron_data": trihedron_data if trihedron_data else None,
        "estimated_volume_cm3": round(volume_cm3, 2),
        "estimated_volume_m3": volume_m3,
        "clusters": clusters if strategy == "svdcodes" else None,
    })


# ── Update mask (GrabCut interactive refinement) ──────────────────────────────

class UpdateMaskRequest(BaseModel):
    run_id: str
    view: str  # 'xy', 'yz', or 'xz'
    focal_length: float
    arm_length_cm: Optional[float] = 20.0
    fg_rects: List[List[int]] = []
    bg_rects: List[List[int]] = []


@app.post("/api/update-mask")
async def update_mask(req: UpdateMaskRequest):
    run_dir = VOLUME_DIR / req.run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    img_path = run_dir / f"photo_{req.view}.jpg"
    mask_path = run_dir / f"mask_{req.view}.png"

    strategy = GrabCutStrategy()
    mask = strategy.create_mask(
        str(img_path),
        resolution=RESOLUTION,
        fg_rects=req.fg_rects,
        bg_rects=req.bg_rects,
    )
    save_mask(mask, mask_path)

    masks = {}
    for v in ["xy", "yz", "xz"]:
        img = Image.open(run_dir / f"mask_{v}.png").convert("L")
        masks[v] = np.array(img) > 128

    arm_len = req.arm_length_cm or 20.0
    volume_cm3, volume_m3, _ = _compute_volumes(
        run_dir=run_dir,
        masks=masks,
        focal_length=req.focal_length,
        arm_length_cm=arm_len,
    )

    ts = int(datetime.now().timestamp())
    return JSONResponse({
        "mask_url": f"/data/{req.run_id}/mask_{req.view}.png?t={ts}",
        "estimated_volume_cm3": round(volume_cm3, 2),
        "estimated_volume_m3": volume_m3,
    })


# ── Resegment (switch algorithm or parameters) ────────────────────────────────

class ResegmentRequest(BaseModel):
    run_id: str
    strategy: str = "trihedron"
    focal_length: float = 50.0
    arm_length_cm: float = 20.0
    alpha: int = 16
    min_cluster_pct: float = 1.0


@app.post("/api/resegment")
async def resegment_images(req: ResegmentRequest):
    run_dir = VOLUME_DIR / req.run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    photo_paths = {
        v: run_dir / f"photo_{v}.jpg" for v in ["xy", "yz", "xz"]
    }
    for v, p in photo_paths.items():
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Photo for view '{v}' not found")

    strategy_obj = _make_strategy(req.strategy)
    masks: Dict[str, np.ndarray] = {}
    clusters: Dict[str, list] = {}
    overlays: Dict[str, str] = {}
    trihedron_data: Dict[str, Any] = {}

    for view, img_path in photo_paths.items():
        if req.strategy == "svdcodes":
            view_clusters = _run_svd_clusters(
                img_path, req.run_id, run_dir, view, RESOLUTION, req.alpha, req.min_cluster_pct
            )
            clusters[view] = view_clusters
            visible_codes = [c["code"] for c in view_clusters]
            masks[view] = _initial_svd_mask(run_dir, view, RESOLUTION, visible_codes)
        elif req.strategy == "trihedron":
            masks[view] = strategy_obj.create_mask(str(img_path), RESOLUTION)
            img_bgr = cv2.imread(str(img_path))
            kp = find_trihedron_keypoints(img_bgr, view=view, arm_length_cm=req.arm_length_cm)
            if kp is not None:
                overlay_bgr = draw_trihedron_overlay(img_bgr, kp)
                overlay_file = run_dir / f"overlay_{view}.jpg"
                cv2.imwrite(str(overlay_file), overlay_bgr)
                overlays[view] = f"/data/{req.run_id}/overlay_{view}.jpg"
                trihedron_data[view] = {
                    "vertex": kp.vertex,
                    "arm_length_px": kp.arm_length_px,
                    "scale_px_per_cm": kp.px_per_cm,
                    "reproj_error_px": kp.reproj_error,
                }
        else:
            is_contrast = view != "xz"
            masks[view] = strategy_obj.create_mask(
                str(img_path), RESOLUTION, is_contrasting_bg=is_contrast
            )

    if req.strategy == "svdcodes":
        clusters_path = run_dir / "svd_clusters.json"
        with open(clusters_path, "w") as f:
            json.dump({"alpha": req.alpha, "min_cluster_pct": req.min_cluster_pct, "clusters": clusters}, f)

    for v, mask in masks.items():
        save_mask(mask, run_dir / f"mask_{v}.png")

    volume_cm3, volume_m3, recon_meta = _compute_volumes(
        run_dir=run_dir,
        masks=masks,
        focal_length=req.focal_length,
        arm_length_cm=req.arm_length_cm,
    )

    if req.strategy == "trihedron" and trihedron_data:
        tri_meta_file = run_dir / "trihedron_metadata.json"
        with open(tri_meta_file, "w") as f:
            json.dump({
                "arm_length_cm": req.arm_length_cm,
                "trihedron_data": trihedron_data,
                "recon_meta": recon_meta,
                "volume_cm3": volume_cm3,
                "volume_m3": volume_m3,
            }, f, indent=2)

    ts = int(datetime.now().timestamp())
    return JSONResponse({
        "masks": {v: f"/data/{req.run_id}/mask_{v}.png?t={ts}" for v in masks},
        "overlays": overlays if overlays else None,
        "trihedron_data": trihedron_data if trihedron_data else None,
        "estimated_volume_cm3": round(volume_cm3, 2),
        "estimated_volume_m3": volume_m3,
        "clusters": clusters if req.strategy == "svdcodes" else None,
    })


# ── SVD Clusters (on-demand refresh) ─────────────────────────────────────────

class SVDClustersRequest(BaseModel):
    run_id: str
    alpha: int = 16
    min_cluster_pct: float = 1.0


@app.post("/api/svd-clusters")
async def svd_clusters(req: SVDClustersRequest):
    run_dir = VOLUME_DIR / req.run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    clusters: Dict[str, list] = {}
    for view in ["xy", "yz", "xz"]:
        img_path = run_dir / f"photo_{view}.jpg"
        if not img_path.exists():
            raise HTTPException(status_code=404, detail=f"Photo for view '{view}' not found")
        clusters[view] = _run_svd_clusters(
            img_path, req.run_id, run_dir, view,
            RESOLUTION, req.alpha, req.min_cluster_pct,
        )

    clusters_path = run_dir / "svd_clusters.json"
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
    arm_length_cm: Optional[float] = 20.0
    selected_codes: Dict[str, List[int]]


@app.post("/api/compute-from-clusters")
async def compute_from_clusters(req: ComputeFromClustersRequest):
    run_dir = VOLUME_DIR / req.run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    masks: Dict[str, np.ndarray] = {}
    for view in ["xy", "yz", "xz"]:
        codes_path = run_dir / f"svd_codes_{view}.npy"
        if not codes_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"SVD codes for view '{view}' not found. Run the SVDCodes algorithm first.",
            )

        codes = np.load(codes_path)
        selected = req.selected_codes.get(view, [])
        if not selected:
            combined = np.zeros(RESOLUTION * RESOLUTION, dtype=bool)
        else:
            combined = np.zeros(RESOLUTION * RESOLUTION, dtype=bool)
            for code in selected:
                combined |= codes == code

        masks[view] = combined.reshape(RESOLUTION, RESOLUTION)
        save_mask(masks[view], run_dir / f"mask_{view}.png")

    arm_len = req.arm_length_cm or 20.0
    volume_cm3, volume_m3, _ = _compute_volumes(
        run_dir=run_dir,
        masks=masks,
        focal_length=req.focal_length,
        arm_length_cm=arm_len,
    )

    ts = int(datetime.now().timestamp())
    return JSONResponse({
        "masks": {v: f"/data/{req.run_id}/mask_{v}.png?t={ts}" for v in masks},
        "estimated_volume_cm3": round(volume_cm3, 2),
        "estimated_volume_m3": volume_m3,
    })


# ── Calculate Volume ─────────────────────────────────────────────────────────

class CalculateVolumeRequest(BaseModel):
    run_id: str
    focal_length: float
    arm_length_cm: Optional[float] = 20.0


@app.post("/api/calculate-volume")
async def calculate_volume(req: CalculateVolumeRequest):
    run_dir = VOLUME_DIR / req.run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    masks = {}
    for view in ["xy", "yz", "xz"]:
        mask_path = run_dir / f"mask_{view}.png"
        if not mask_path.exists():
            raise HTTPException(status_code=404, detail=f"Mask for {view} not found")
        img = Image.open(mask_path).convert("L")
        masks[view] = np.array(img) > 128

    arm_len = req.arm_length_cm or 20.0
    volume_cm3, volume_m3, _ = _compute_volumes(
        run_dir=run_dir,
        masks=masks,
        focal_length=req.focal_length,
        arm_length_cm=arm_len,
    )

    return JSONResponse({
        "estimated_volume_cm3": round(volume_cm3, 2),
        "estimated_volume_m3": volume_m3,
    })
