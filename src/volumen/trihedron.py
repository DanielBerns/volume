"""
Module for detection, localization, and camera calibration using a red trihedron reference frame.
"""
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "TrihedronKeypoints",
    "extract_red_mask",
    "find_trihedron_keypoints",
    "draw_trihedron_overlay",
]

@dataclass
class TrihedronKeypoints:
    vertex: Tuple[float, float]               # (u, v) in pixels
    tip_x: Tuple[float, float]                # (u, v) in pixels for X arm
    tip_y: Tuple[float, float]                # (u, v) in pixels for Y arm
    tip_z: Tuple[float, float]                # (u, v) in pixels for Z arm
    arm_length_px: float                      # Average projected length of in-plane arms (px)
    px_per_cm: float                          # Metric scale: pixels per cm
    mask: Optional[np.ndarray] = None         # Binary uint8 mask of the trihedron (0 or 255)
    rvec: Optional[np.ndarray] = None         # PnP rotation vector (3, 1)
    tvec: Optional[np.ndarray] = None         # PnP translation vector (3, 1)
    reproj_error: Optional[float] = None      # Mean reprojection error (px)
    arm_width_px: Optional[float] = None      # Measured projected width of trihedron arms (px)
    image_shape: Optional[Tuple[int, int]] = None  # Source image (height, width) in pixels

def extract_red_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Extracts binary mask of the red trihedron in BGR image."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 70, 35]), np.array([15, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165, 70, 35]), np.array([180, 255, 255]))
    red_raw = mask1 | mask2

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    red_clean = cv2.morphologyEx(red_raw, cv2.MORPH_OPEN, kernel)
    red_clean = cv2.morphologyEx(red_clean, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(red_clean)
    if num_labels <= 1:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = np.max(areas)

    tri_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 0.05 * max_area:
            tri_mask[labels == i] = 255

    return tri_mask


def _median_tip(xs: np.ndarray, ys: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """Return the median (x, y) coordinates of pixels satisfying mask."""
    return (float(np.median(xs[mask])), float(np.median(ys[mask])))


def _fit_arm_line_and_width(
    xs: np.ndarray,
    ys: np.ndarray,
    horizontal: bool,
) -> Tuple[Optional[Tuple[float, float, float, float]], Optional[float]]:
    """
    Fit a 2D line (vx, vy, x0, y0) and estimate median cross-sectional width
    for a set of pixels along a single trihedron arm.
    """
    if len(xs) < 10:
        return None, None

    pts = np.column_stack([xs, ys]).astype(np.float32)
    line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    vx, vy, x0, y0 = float(line[0]), float(line[1]), float(line[2]), float(line[3])

    widths = []
    if horizontal:
        for ux in np.unique(xs):
            col_ys = ys[xs == ux]
            widths.append(float(np.max(col_ys) - np.min(col_ys) + 1))
    else:
        for uy in np.unique(ys):
            row_xs = xs[ys == uy]
            widths.append(float(np.max(row_xs) - np.min(row_xs) + 1))

    arm_w = float(np.median(widths)) if widths else None
    return (vx, vy, x0, y0), arm_w


def _intersect_lines(
    line_a: Tuple[float, float, float, float],
    line_b: Tuple[float, float, float, float],
) -> Optional[Tuple[float, float]]:
    """Compute intersection of two lines parameterized as (vx, vy, x0, y0)."""
    vx1, vy1, x1, y1 = line_a
    vx2, vy2, x2, y2 = line_b
    det = -vx1 * vy2 + vy1 * vx2
    if abs(det) < 1e-6:
        return None
    dx = x2 - x1
    dy = y2 - y1
    t = (-dx * vy2 + dy * vx2) / det
    return (x1 + t * vx1, y1 + t * vy1)


def _refine_corner_from_arms(
    xs: np.ndarray,
    ys: np.ndarray,
    tip_h: Tuple[float, float],
    tip_v: Tuple[float, float],
    v_goes_down: bool,
) -> Tuple[Tuple[float, float], Optional[float]]:
    """
    Refine corner vertex p0 and estimate arm width (px) by fitting centerlines
    to the outer sections of the horizontal and vertical in-plane arms.
    """
    c0 = (tip_v[0], tip_h[1])
    span_h = abs(tip_h[0] - c0[0])
    span_v = abs(tip_v[1] - c0[1])

    band_h = max(12.0, 0.25 * span_h)
    band_v = max(12.0, 0.25 * span_v)

    h_mask = (
        (xs >= c0[0] + 0.35 * span_h)
        & (xs <= c0[0] + 0.92 * span_h)
        & (np.abs(ys - tip_h[1]) <= band_h)
    )
    if v_goes_down:
        v_mask = (
            (ys >= c0[1] + 0.35 * span_v)
            & (ys <= c0[1] + 0.92 * span_v)
            & (np.abs(xs - tip_v[0]) <= band_v)
        )
    else:
        v_mask = (
            (ys <= c0[1] - 0.35 * span_v)
            & (ys >= c0[1] - 0.92 * span_v)
            & (np.abs(xs - tip_v[0]) <= band_v)
        )

    line_h, w_h = _fit_arm_line_and_width(xs[h_mask], ys[h_mask], horizontal=True)
    line_v, w_v = _fit_arm_line_and_width(xs[v_mask], ys[v_mask], horizontal=False)

    widths = [w for w in (w_h, w_v) if w is not None and w > 0]
    arm_width_px = float(np.median(widths)) if widths else None

    if line_h is not None and line_v is not None:
        inter = _intersect_lines(line_h, line_v)
        if inter is not None:
            max_shift = max(40.0, 0.25 * max(span_h, span_v))
            if np.hypot(inter[0] - c0[0], inter[1] - c0[1]) <= max_shift:
                return inter, arm_width_px

    search_r = max(15.0, min(60.0, 0.1 * max(span_h, span_v)))
    dists = np.hypot(xs - c0[0], ys - c0[1])
    nearby = dists < search_r
    if np.any(nearby):
        p0 = (float(np.mean(xs[nearby])), float(np.mean(ys[nearby])))
    else:
        p0 = c0
    return p0, arm_width_px


def find_trihedron_keypoints(
    image_bgr: np.ndarray,
    view: str,
    arm_length_cm: float = 20.0,
    arm_width_cm: float = 0.5,
    focal_length_mm: float = 50.0,
    sensor_width_mm: float = 36.0,
) -> Optional[TrihedronKeypoints]:
    """
    Detects the red trihedron in the image for the given view ('xy', 'xz', 'yz').
    Default reference dimensions: 200 mm (20.0 cm) arm length, 5 mm (0.5 cm) arm width.
    Returns TrihedronKeypoints with vertex, arm endpoints, metric scale, and PnP pose.
    """
    height, width = image_bgr.shape[:2]
    tri_mask = extract_red_mask(image_bgr)
    min_area = min(5000, max(200, int(0.002 * height * width)))
    if np.sum(tri_mask > 0) < min_area:
        logger.warning("Trihedron mask area is too small (<%d px)", min_area)
        return None

    ys, xs = np.where(tri_mask > 0)
    x_span = float(np.max(xs) - np.min(xs))
    y_span = float(np.max(ys) - np.min(ys))
    slice_x = max(2.0, 0.02 * x_span)
    slice_y = max(2.0, 0.02 * y_span)

    if view == "xy":
        # In XY (top-down view): PX rightmost (+x), PY bottommost (+y), PZ toward camera
        tip_x = _median_tip(xs, ys, xs >= np.max(xs) - slice_x)
        tip_y = _median_tip(xs, ys, ys >= np.max(ys) - slice_y)
        min_idx = np.argmin(xs + ys)
        tip_z = (float(xs[min_idx]), float(ys[min_idx]))

        p0, arm_width_px = _refine_corner_from_arms(xs, ys, tip_h=tip_x, tip_v=tip_y, v_goes_down=True)
        len_x = np.hypot(tip_x[0] - p0[0], tip_x[1] - p0[1])
        len_y = np.hypot(tip_y[0] - p0[0], tip_y[1] - p0[1])
        avg_arm_len_px = (len_x + len_y) / 2.0

    elif view == "xz":
        # In XZ (side elevation along Y): PZ vertical up (-y), PX horizontal right (+x), PY forward
        tip_z = _median_tip(xs, ys, ys <= np.min(ys) + slice_y)
        tip_x = _median_tip(xs, ys, xs >= np.max(xs) - slice_x)
        bot_left_idx = np.argmin(xs)
        tip_y = (float(xs[bot_left_idx]), float(ys[bot_left_idx]))

        p0, arm_width_px = _refine_corner_from_arms(xs, ys, tip_h=tip_x, tip_v=tip_z, v_goes_down=False)
        len_x = np.hypot(tip_x[0] - p0[0], tip_x[1] - p0[1])
        len_z = np.hypot(tip_z[0] - p0[0], tip_z[1] - p0[1])
        avg_arm_len_px = (len_x + len_z) / 2.0

    elif view == "yz":
        # In YZ (side elevation along X): PZ vertical up (-y), PY horizontal right (+x), PX forward
        tip_z = _median_tip(xs, ys, ys <= np.min(ys) + slice_y)
        tip_y = _median_tip(xs, ys, xs >= np.max(xs) - slice_x)
        bot_left_idx = np.argmin(xs)
        tip_x = (float(xs[bot_left_idx]), float(ys[bot_left_idx]))

        p0, arm_width_px = _refine_corner_from_arms(xs, ys, tip_h=tip_y, tip_v=tip_z, v_goes_down=False)
        len_y = np.hypot(tip_y[0] - p0[0], tip_y[1] - p0[1])
        len_z = np.hypot(tip_z[0] - p0[0], tip_z[1] - p0[1])
        avg_arm_len_px = (len_y + len_z) / 2.0
    else:
        raise ValueError(f"Unknown view: {view}")

    px_per_cm = avg_arm_len_px / arm_length_cm

    # 3D points in cm:
    obj_pts = np.array([
        [0.0, 0.0, 0.0],
        [arm_length_cm, 0.0, 0.0],
        [0.0, arm_length_cm, 0.0],
        [0.0, 0.0, arm_length_cm]
    ], dtype=np.float32)

    img_pts = np.array([p0, tip_x, tip_y, tip_z], dtype=np.float32)

    f_px = focal_length_mm * (width / sensor_width_mm)
    K = np.array([
        [f_px, 0.0, width / 2.0],
        [0.0, f_px, height / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    dist_coeffs = np.zeros(4, dtype=np.float32)

    rvec, tvec, reproj_err = None, None, None
    try:
        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
        if success:
            rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, dist_coeffs, rvec, tvec)
            proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist_coeffs)
            reproj_err = float(np.mean(np.linalg.norm(img_pts - proj_pts.reshape(-1, 2), axis=1)))
    except Exception as e:
        logger.warning("PnP solve failed for view %s: %s", view, e)

    return TrihedronKeypoints(
        vertex=p0,
        tip_x=tip_x,
        tip_y=tip_y,
        tip_z=tip_z,
        arm_length_px=avg_arm_len_px,
        px_per_cm=px_per_cm,
        mask=tri_mask,
        rvec=rvec,
        tvec=tvec,
        reproj_error=reproj_err,
        arm_width_px=arm_width_px,
        image_shape=(height, width),
    )

def draw_trihedron_overlay(image_bgr: np.ndarray, kp: TrihedronKeypoints) -> np.ndarray:
    """
    Renders an annotated visualization overlay with coordinate axes and labels:
    - X Axis: Red line
    - Y Axis: Green line
    - Z Axis: Blue line
    - Origin Vertex: Yellow circle
    """
    vis = image_bgr.copy()
    v = (int(round(kp.vertex[0])), int(round(kp.vertex[1])))
    tx = (int(round(kp.tip_x[0])), int(round(kp.tip_x[1])))
    ty = (int(round(kp.tip_y[0])), int(round(kp.tip_y[1])))
    tz = (int(round(kp.tip_z[0])), int(round(kp.tip_z[1])))

    # Highlight arms
    cv2.line(vis, v, tx, (0, 0, 255), 4)    # Red: X axis
    cv2.line(vis, v, ty, (0, 255, 0), 4)    # Green: Y axis
    cv2.line(vis, v, tz, (255, 0, 0), 4)    # Blue: Z axis

    # Highlight origin and tips
    cv2.circle(vis, v, 8, (0, 255, 255), -1)  # Yellow: Origin
    cv2.circle(vis, tx, 6, (0, 0, 255), -1)   # Red: Tip X
    cv2.circle(vis, ty, 6, (0, 255, 0), -1)   # Green: Tip Y
    cv2.circle(vis, tz, 6, (255, 0, 0), -1)   # Blue: Tip Z

    # Text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis, "O", (v[0] + 10, v[1] - 10), font, 0.8, (0, 255, 255), 2)
    cv2.putText(vis, "X", (tx[0] + 10, tx[1]), font, 0.8, (0, 0, 255), 2)
    cv2.putText(vis, "Y", (ty[0] + 10, ty[1]), font, 0.8, (0, 255, 0), 2)
    cv2.putText(vis, "Z", (tz[0] + 10, tz[1]), font, 0.8, (255, 0, 0), 2)

    return vis
