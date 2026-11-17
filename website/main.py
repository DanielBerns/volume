import csv
import io
import json
import logging
import math
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field
from scipy import stats

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
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@app.get("/")
async def read_index():
    return FileResponse(STATIC_DIR / "index.html")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_run_dir(run_id: str, must_exist: bool = True) -> Path:
    """Validate run_id and return the resolved directory inside VOLUME_DIR."""
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=400, detail="Invalid run_id")
    base = VOLUME_DIR.resolve()
    run_dir = (base / run_id).resolve()
    if run_dir.parent != base:
        raise HTTPException(status_code=400, detail="Invalid run_id")
    if must_exist and not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    return run_dir


def _normalize_float(val: Any) -> Optional[float]:
    """Convert numeric value to finite float without negative zero."""
    if val is None:
        return None
    f = float(val)
    if not math.isfinite(f):
        return None
    return 0.0 if f == 0.0 else f


def _clean_optional_text(val: Optional[str]) -> Optional[str]:
    """Strip optional string fields and return None when empty."""
    if val is None:
        return None
    cleaned = val.strip()
    return cleaned if cleaned else None


def _validate_positive_volume(val: Optional[float], field_name: str = "direct_volume_cm3") -> Optional[float]:
    """Ensure optional direct volume is finite and strictly positive."""
    if val is None:
        return None
    if not math.isfinite(val) or val <= 0:
        raise HTTPException(status_code=422, detail=f"{field_name} must be a finite positive number")
    return float(val)


def _load_run_measurement(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load run measurement metadata, falling back to trihedron_metadata.json."""
    meas_file = run_dir / "measurement.json"
    if meas_file.exists():
        data = json.loads(meas_file.read_text(encoding="utf-8"))
        data["run_id"] = run_dir.name
        data["estimated_volume_cm3"] = _normalize_float(data.get("estimated_volume_cm3"))
        data["estimated_volume_m3"] = _normalize_float(data.get("estimated_volume_m3"))
        data["direct_volume_cm3"] = _normalize_float(data.get("direct_volume_cm3"))
        return data

    tri_file = run_dir / "trihedron_metadata.json"
    if tri_file.exists():
        tri = json.loads(tri_file.read_text(encoding="utf-8"))
        vol_cm3 = _normalize_float(tri.get("volume_cm3"))
        vol_m3 = _normalize_float(tri.get("volume_m3"))
        if vol_cm3 is None and vol_m3 is not None:
            vol_cm3 = _normalize_float(vol_m3 * 1e6)
        if vol_m3 is None and vol_cm3 is not None:
            vol_m3 = _normalize_float(vol_cm3 / 1e6)
        return {
            "run_id": run_dir.name,
            "sample_name": None,
            "strategy": "trihedron",
            "estimated_volume_cm3": round(vol_cm3, 2) if vol_cm3 is not None else None,
            "estimated_volume_m3": vol_m3,
            "direct_volume_cm3": None,
            "direct_method": None,
            "created_at": None,
            "updated_at": None,
        }

    return None


def _save_run_measurement(run_dir: Path, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Merge updates into run_dir/measurement.json and return the saved record."""
    now_iso = datetime.now().isoformat(timespec="seconds")
    record = _load_run_measurement(run_dir) or {
        "run_id": run_dir.name,
        "sample_name": None,
        "strategy": None,
        "estimated_volume_cm3": None,
        "estimated_volume_m3": None,
        "direct_volume_cm3": None,
        "direct_method": None,
        "created_at": now_iso,
        "updated_at": now_iso,
    }
    if record.get("created_at") is None:
        record["created_at"] = now_iso

    record.update(updates)
    record["estimated_volume_cm3"] = _normalize_float(record.get("estimated_volume_cm3"))
    record["estimated_volume_m3"] = _normalize_float(record.get("estimated_volume_m3"))
    record["direct_volume_cm3"] = _normalize_float(record.get("direct_volume_cm3"))
    record["updated_at"] = now_iso

    meas_file = run_dir / "measurement.json"
    meas_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _collect_all_runs() -> List[Dict[str, Any]]:
    """Collect all run records in VOLUME_DIR that have an estimated volume."""
    if not VOLUME_DIR.exists():
        return []
    runs: List[Dict[str, Any]] = []
    for entry in sorted(VOLUME_DIR.iterdir(), key=lambda p: p.name, reverse=True):
        if not entry.is_dir():
            continue
        meas = _load_run_measurement(entry)
        if meas is not None and meas.get("estimated_volume_cm3") is not None:
            runs.append(meas)
    return runs


def _finite_or_none(val: float) -> Optional[float]:
    return float(val) if math.isfinite(val) else None


def _compute_correlation(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute paired differences, regression fit, and correlation statistics."""
    paired: List[Dict[str, Any]] = []
    for item in runs:
        est = item.get("estimated_volume_cm3")
        direct = item.get("direct_volume_cm3")
        if est is None or direct is None or direct <= 0:
            continue
        err = est - direct
        rel_err = (err / direct) * 100.0
        paired.append({
            **item,
            "error_cm3": round(err, 4),
            "abs_error_cm3": round(abs(err), 4),
            "rel_error_pct": round(rel_err, 2),
        })

    n = len(paired)
    pearson_r: Optional[float] = None
    pearson_p: Optional[float] = None
    r_squared: Optional[float] = None
    spearman_rho: Optional[float] = None
    spearman_p: Optional[float] = None
    slope: Optional[float] = None
    intercept: Optional[float] = None
    mae_cm3: Optional[float] = None
    rmse_cm3: Optional[float] = None
    mean_bias_cm3: Optional[float] = None
    mape_pct: Optional[float] = None

    if n >= 1:
        x = np.array([p["direct_volume_cm3"] for p in paired], dtype=float)
        y = np.array([p["estimated_volume_cm3"] for p in paired], dtype=float)
        diffs = y - x
        mae_cm3 = _finite_or_none(float(np.mean(np.abs(diffs))))
        rmse_cm3 = _finite_or_none(float(np.sqrt(np.mean(diffs ** 2))))
        mean_bias_cm3 = _finite_or_none(float(np.mean(diffs)))
        mape_pct = _finite_or_none(float(np.mean(np.abs(diffs) / x) * 100.0))

        if n >= 2 and float(np.std(x)) > 0.0:
            lr = stats.linregress(x, y)
            slope = _finite_or_none(float(lr.slope))
            intercept = _finite_or_none(float(lr.intercept))

            if float(np.std(y)) > 0.0:
                pr = stats.pearsonr(x, y)
                pearson_r = _finite_or_none(float(pr.statistic))
                pearson_p = _finite_or_none(float(pr.pvalue))
                if pearson_r is not None:
                    r_squared = _finite_or_none(pearson_r ** 2)

                sr = stats.spearmanr(x, y)
                spearman_rho = _finite_or_none(float(sr.statistic))
                spearman_p = _finite_or_none(float(sr.pvalue))

    return {
        "runs": runs,
        "paired": paired,
        "stats": {
            "n": n,
            "pearson_r": round(pearson_r, 4) if pearson_r is not None else None,
            "pearson_p": round(pearson_p, 6) if pearson_p is not None else None,
            "r_squared": round(r_squared, 4) if r_squared is not None else None,
            "spearman_rho": round(spearman_rho, 4) if spearman_rho is not None else None,
            "spearman_p": round(spearman_p, 6) if spearman_p is not None else None,
            "slope": round(slope, 4) if slope is not None else None,
            "intercept": round(intercept, 4) if intercept is not None else None,
            "mae_cm3": round(mae_cm3, 2) if mae_cm3 is not None else None,
            "rmse_cm3": round(rmse_cm3, 2) if rmse_cm3 is not None else None,
            "mean_bias_cm3": round(mean_bias_cm3, 2) if mean_bias_cm3 is not None else None,
            "mape_pct": round(mape_pct, 2) if mape_pct is not None else None,
        },
    }


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
    direct_volume_cm3: Optional[float] = Form(None),
    sample_name: Optional[str] = Form(None),
    direct_method: Optional[str] = Form(None),
):
    validated_direct = _validate_positive_volume(direct_volume_cm3)
    cleaned_sample = _clean_optional_text(sample_name)
    cleaned_method = _clean_optional_text(direct_method)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_dir = VOLUME_DIR / run_id
    suffix = 1
    while run_dir.exists():
        run_id = f"run_{timestamp}_{suffix}"
        run_dir = VOLUME_DIR / run_id
        suffix += 1
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
    est_cm3 = _normalize_float(round(volume_cm3, 2))
    est_m3 = _normalize_float(volume_m3)

    # Save trihedron metadata to run_dir
    tri_meta_file = run_dir / "trihedron_metadata.json"
    with open(tri_meta_file, "w") as f:
        json.dump({
            "arm_length_cm": arm_length_cm,
            "trihedron_data": trihedron_data,
            "recon_meta": recon_meta,
            "volume_cm3": est_cm3,
            "volume_m3": est_m3,
        }, f, indent=2)

    _save_run_measurement(run_dir, {
        "strategy": strategy,
        "sample_name": cleaned_sample,
        "estimated_volume_cm3": est_cm3,
        "estimated_volume_m3": est_m3,
        "direct_volume_cm3": validated_direct,
        "direct_method": cleaned_method,
    })

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
        "estimated_volume_cm3": est_cm3,
        "estimated_volume_m3": est_m3,
        "direct_volume_cm3": validated_direct,
        "sample_name": cleaned_sample,
        "direct_method": cleaned_method,
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
    run_dir = _resolve_run_dir(req.run_id)

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
    est_cm3 = _normalize_float(round(volume_cm3, 2))
    est_m3 = _normalize_float(volume_m3)
    _save_run_measurement(run_dir, {
        "estimated_volume_cm3": est_cm3,
        "estimated_volume_m3": est_m3,
    })

    ts = int(datetime.now().timestamp())
    return JSONResponse({
        "mask_url": f"/data/{req.run_id}/mask_{req.view}.png?t={ts}",
        "estimated_volume_cm3": est_cm3,
        "estimated_volume_m3": est_m3,
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
    run_dir = _resolve_run_dir(req.run_id)

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
    est_cm3 = _normalize_float(round(volume_cm3, 2))
    est_m3 = _normalize_float(volume_m3)

    if req.strategy == "trihedron" and trihedron_data:
        tri_meta_file = run_dir / "trihedron_metadata.json"
        with open(tri_meta_file, "w") as f:
            json.dump({
                "arm_length_cm": req.arm_length_cm,
                "trihedron_data": trihedron_data,
                "recon_meta": recon_meta,
                "volume_cm3": est_cm3,
                "volume_m3": est_m3,
            }, f, indent=2)

    _save_run_measurement(run_dir, {
        "strategy": req.strategy,
        "estimated_volume_cm3": est_cm3,
        "estimated_volume_m3": est_m3,
    })

    ts = int(datetime.now().timestamp())
    return JSONResponse({
        "masks": {v: f"/data/{req.run_id}/mask_{v}.png?t={ts}" for v in masks},
        "overlays": overlays if overlays else None,
        "trihedron_data": trihedron_data if trihedron_data else None,
        "estimated_volume_cm3": est_cm3,
        "estimated_volume_m3": est_m3,
        "clusters": clusters if req.strategy == "svdcodes" else None,
    })


# ── SVD Clusters (on-demand refresh) ─────────────────────────────────────────

class SVDClustersRequest(BaseModel):
    run_id: str
    alpha: int = 16
    min_cluster_pct: float = 1.0


@app.post("/api/svd-clusters")
async def svd_clusters(req: SVDClustersRequest):
    run_dir = _resolve_run_dir(req.run_id)

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
    run_dir = _resolve_run_dir(req.run_id)

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
    est_cm3 = _normalize_float(round(volume_cm3, 2))
    est_m3 = _normalize_float(volume_m3)
    _save_run_measurement(run_dir, {
        "strategy": "svdcodes",
        "estimated_volume_cm3": est_cm3,
        "estimated_volume_m3": est_m3,
    })

    ts = int(datetime.now().timestamp())
    return JSONResponse({
        "masks": {v: f"/data/{req.run_id}/mask_{v}.png?t={ts}" for v in masks},
        "estimated_volume_cm3": est_cm3,
        "estimated_volume_m3": est_m3,
    })


# ── Calculate Volume ─────────────────────────────────────────────────────────

class CalculateVolumeRequest(BaseModel):
    run_id: str
    focal_length: float
    arm_length_cm: Optional[float] = 20.0


@app.post("/api/calculate-volume")
async def calculate_volume(req: CalculateVolumeRequest):
    run_dir = _resolve_run_dir(req.run_id)

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
    est_cm3 = _normalize_float(round(volume_cm3, 2))
    est_m3 = _normalize_float(volume_m3)
    _save_run_measurement(run_dir, {
        "estimated_volume_cm3": est_cm3,
        "estimated_volume_m3": est_m3,
    })

    return JSONResponse({
        "estimated_volume_cm3": est_cm3,
        "estimated_volume_m3": est_m3,
    })


# ── Direct Volume Measurements & Correlation Study ────────────────────────────

class DirectVolumeRequest(BaseModel):
    run_id: str
    direct_volume_cm3: float = Field(..., gt=0, allow_inf_nan=False)
    sample_name: Optional[str] = None
    direct_method: Optional[str] = None


@app.post("/api/direct-volume")
async def save_direct_volume(req: DirectVolumeRequest):
    run_dir = _resolve_run_dir(req.run_id)
    existing = _load_run_measurement(run_dir)
    if existing is None or existing.get("estimated_volume_cm3") is None:
        raise HTTPException(status_code=400, detail="Run has no computed volume estimate yet")

    validated_direct = _validate_positive_volume(req.direct_volume_cm3)
    updates: Dict[str, Any] = {
        "direct_volume_cm3": validated_direct,
    }
    if req.sample_name is not None:
        updates["sample_name"] = _clean_optional_text(req.sample_name)
    if req.direct_method is not None:
        updates["direct_method"] = _clean_optional_text(req.direct_method)

    saved = _save_run_measurement(run_dir, updates)
    correlation = _compute_correlation(_collect_all_runs())
    return JSONResponse({
        "measurement": saved,
        "correlation": correlation,
    })


@app.delete("/api/direct-volume/{run_id}")
async def delete_direct_volume(run_id: str):
    run_dir = _resolve_run_dir(run_id)
    existing = _load_run_measurement(run_dir)
    if existing is None:
        raise HTTPException(status_code=404, detail="Measurement metadata not found")

    _save_run_measurement(run_dir, {
        "direct_volume_cm3": None,
        "direct_method": None,
    })
    correlation = _compute_correlation(_collect_all_runs())
    return JSONResponse({
        "run_id": run_id,
        "correlation": correlation,
    })


@app.get("/api/correlation")
async def get_correlation():
    correlation = _compute_correlation(_collect_all_runs())
    return JSONResponse(correlation)


@app.get("/api/correlation/export")
async def export_correlation_csv():
    correlation = _compute_correlation(_collect_all_runs())
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "run_id",
        "sample_name",
        "strategy",
        "direct_volume_cm3",
        "estimated_volume_cm3",
        "error_cm3",
        "rel_error_pct",
        "direct_method",
        "updated_at",
    ])
    for row in correlation["paired"]:
        writer.writerow([
            row.get("run_id") or "",
            row.get("sample_name") or "",
            row.get("strategy") or "",
            row.get("direct_volume_cm3"),
            row.get("estimated_volume_cm3"),
            row.get("error_cm3"),
            row.get("rel_error_pct"),
            row.get("direct_method") or "",
            row.get("updated_at") or "",
        ])

    return Response(
        content=output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="correlation_study.csv"'},
    )
