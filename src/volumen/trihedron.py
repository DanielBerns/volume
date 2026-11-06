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

def find_trihedron_keypoints(
    image_bgr: np.ndarray,
    view: str,
    arm_length_cm: float = 20.0,
    focal_length_mm: float = 50.0,
    sensor_width_mm: float = 36.0,
) -> Optional[TrihedronKeypoints]:
    """
    Detects the red trihedron in the image for the given view ('xy', 'xz', 'yz').
    Returns TrihedronKeypoints with vertex, arm endpoints, metric scale, and PnP pose.
    """
    height, width = image_bgr.shape[:2]
    tri_mask = extract_red_mask(image_bgr)
    min_area = min(5000, max(200, int(0.002 * height * width)))
    if np.sum(tri_mask > 0) < min_area:
        logger.warning("Trihedron mask area is too small (<%d px)", min_area)
        return None

    ys, xs = np.where(tri_mask > 0)

    if view == "xy":
        # In XY (top-down view):
        # PX is rightmost (max X) along table
        # PY is bottommost (max Y) along table
        # PZ is pointing toward camera (up-left: min X + Y)
        # P0 corner is junction between PX and PY (near (PY[0], PX[1]))
        tip_x = (float(xs[np.argmax(xs)]), float(ys[np.argmax(xs)]))
        tip_y = (float(xs[np.argmax(ys)]), float(ys[np.argmax(ys)]))
        min_idx = np.argmin(xs + ys)
        tip_z = (float(xs[min_idx]), float(ys[min_idx]))

        approx_corner = np.array([tip_y[0], tip_x[1]])
        dists = np.hypot(xs - approx_corner[0], ys - approx_corner[1])
        nearby = dists < 40
        if np.any(nearby):
            p0 = (float(np.mean(xs[nearby])), float(np.mean(ys[nearby])))
        else:
            p0 = (float(approx_corner[0]), float(approx_corner[1]))

        len_x = np.hypot(tip_x[0] - p0[0], tip_x[1] - p0[1])
        len_y = np.hypot(tip_y[0] - p0[0], tip_y[1] - p0[1])
        avg_arm_len_px = (len_x + len_y) / 2.0

    elif view == "xz":
        # In XZ (side elevation view along Y):
        # PZ is vertical arm (pointing straight up: min y)
        # PX is horizontal arm on table (pointing right: max x)
        # PY is forward arm (pointing toward camera down-left: min x, max y)
        # P0 is junction between PZ and PX: x near PZ[0], y near PX[1]
        top_idx = np.argmin(ys)
        tip_z = (float(xs[top_idx]), float(ys[top_idx]))

        right_idx = np.argmax(xs)
        tip_x = (float(xs[right_idx]), float(ys[right_idx]))

        bot_left_idx = np.argmin(xs)
        tip_y = (float(xs[bot_left_idx]), float(ys[bot_left_idx]))

        approx_corner = np.array([tip_z[0], tip_x[1]])
        dists = np.hypot(xs - approx_corner[0], ys - approx_corner[1])
        nearby = dists < 40
        if np.any(nearby):
            p0 = (float(np.mean(xs[nearby])), float(np.mean(ys[nearby])))
        else:
            p0 = (float(approx_corner[0]), float(approx_corner[1]))

        len_x = np.hypot(tip_x[0] - p0[0], tip_x[1] - p0[1])
        len_z = np.hypot(tip_z[0] - p0[0], tip_z[1] - p0[1])
        avg_arm_len_px = (len_x + len_z) / 2.0

    elif view == "yz":
        # In YZ (side elevation view along X):
        # PZ is vertical arm (pointing straight up: min y)
        # PY is horizontal arm on table (pointing right: max x)
        # PX is forward arm (pointing toward camera down-left: min x, max y)
        # P0 is junction between PZ and PY
        top_idx = np.argmin(ys)
        tip_z = (float(xs[top_idx]), float(ys[top_idx]))

        right_idx = np.argmax(xs)
        tip_y = (float(xs[right_idx]), float(ys[right_idx]))

        bot_left_idx = np.argmin(xs)
        tip_x = (float(xs[bot_left_idx]), float(ys[bot_left_idx]))

        approx_corner = np.array([tip_z[0], tip_y[1]])
        dists = np.hypot(xs - approx_corner[0], ys - approx_corner[1])
        nearby = dists < 40
        if np.any(nearby):
            p0 = (float(np.mean(xs[nearby])), float(np.mean(ys[nearby])))
        else:
            p0 = (float(approx_corner[0]), float(approx_corner[1]))

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
        reproj_error=reproj_err
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
