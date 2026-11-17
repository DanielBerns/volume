"""
3D visual hull reconstruction algorithms, supporting perspective and trihedron-calibrated reference frames.
"""
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from volumen.trihedron import TrihedronKeypoints

__all__ = [
    "compute_intrinsic_matrix",
    "compute_extrinsic_matrix",
    "perspective_visual_hull",
    "trihedron_visual_hull",
    "projective_bounds_volume",
]

def compute_intrinsic_matrix(focal_length_35mm: float, resolution: int) -> np.ndarray:
    """
    Computes a simple pinhole intrinsic matrix based on 35mm equivalent focal length.
    Assuming square pixels and optical center at the image center.
    """
    sensor_width = 36.0
    f_px = focal_length_35mm * (resolution / sensor_width)
    cx = resolution / 2.0
    cy = resolution / 2.0

    K = np.array([
        [f_px, 0.0, cx],
        [0.0, f_px, cy],
        [0.0, 0.0, 1.0]
    ])
    return K

def compute_extrinsic_matrix(cam_pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Computes a 4x4 view matrix using lookAt method."""
    z_axis = target - cam_pos
    z_axis = z_axis / np.linalg.norm(z_axis)

    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross(np.array([1.0, 0.0, 0.0]), z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    R = np.vstack([x_axis, y_axis, z_axis])
    t = -R @ cam_pos

    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = t
    return E

def perspective_visual_hull(
    mask_xy: np.ndarray,
    mask_yz: np.ndarray,
    mask_xz: np.ndarray,
    focal_length: float = 50.0,
    resolution: int = 256,
    camera_distance: float = 1.5,
    box_size: float = 1.0
) -> float:
    """Computes volume using legacy uncalibrated perspective projection."""
    x = np.linspace(-box_size/2, box_size/2, resolution)
    y = np.linspace(-box_size/2, box_size/2, resolution)
    z = np.linspace(-box_size/2, box_size/2, resolution)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    pts_3d = np.vstack((X.ravel(), Y.ravel(), Z.ravel(), np.ones_like(X.ravel())))

    K = compute_intrinsic_matrix(focal_length, resolution)
    K_proj = np.hstack((K, np.zeros((3, 1))))

    E_xy = compute_extrinsic_matrix(
        cam_pos=np.array([0.0, 0.0, -camera_distance]),
        target=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, -1.0, 0.0])
    )
    P_xy = K_proj @ E_xy

    E_yz = compute_extrinsic_matrix(
        cam_pos=np.array([-camera_distance, 0.0, 0.0]),
        target=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, -1.0, 0.0])
    )
    P_yz = K_proj @ E_yz

    E_xz = compute_extrinsic_matrix(
        cam_pos=np.array([0.0, -camera_distance, 0.0]),
        target=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 0.0, 1.0])
    )
    P_xz = K_proj @ E_xz

    def project_and_sample(P, mask):
        pts_2d = P @ pts_3d
        pts_2d = pts_2d[:2, :] / pts_2d[2, :]
        u = np.round(pts_2d[0, :]).astype(int)
        v = np.round(pts_2d[1, :]).astype(int)
        valid = (u >= 0) & (u < resolution) & (v >= 0) & (v < resolution)
        res = np.zeros_like(valid)
        res[valid] = mask[v[valid], u[valid]]
        return res

    in_xy = project_and_sample(P_xy, mask_xy)
    in_yz = project_and_sample(P_yz, mask_yz)
    in_xz = project_and_sample(P_xz, mask_xz)

    voxel_grid = in_xy & in_yz & in_xz
    filled_voxels = np.sum(voxel_grid)
    voxel_volume_m3 = (box_size / resolution) ** 3
    estimated_volume_m3 = filled_voxels * voxel_volume_m3
    return float(estimated_volume_m3)


def _resolve_view_scale(
    kp: TrihedronKeypoints,
    mask: np.ndarray,
    orig_shape: Optional[Tuple[int, int]],
) -> Tuple[Tuple[float, float], float, float]:
    """Scale trihedron vertex and px/cm metric factors to the mask's pixel grid."""
    h_m, w_m = mask.shape[:2]
    src_shape = orig_shape if orig_shape is not None else kp.image_shape
    if src_shape is not None:
        sx = w_m / float(src_shape[1])
        sy = h_m / float(src_shape[0])
    else:
        sx, sy = 1.0, 1.0

    px_per_cm_x = float(kp.px_per_cm) * sx
    px_per_cm_y = float(kp.px_per_cm) * sy
    if px_per_cm_x <= 0.0 or px_per_cm_y <= 0.0:
        raise ValueError("Trihedron scale px_per_cm must be strictly positive")

    v_scaled = (float(kp.vertex[0]) * sx, float(kp.vertex[1]) * sy)
    return v_scaled, px_per_cm_x, px_per_cm_y


def trihedron_visual_hull(
    mask_xy: np.ndarray,
    mask_yz: np.ndarray,
    mask_xz: np.ndarray,
    kp_xy: TrihedronKeypoints,
    kp_yz: TrihedronKeypoints,
    kp_xz: TrihedronKeypoints,
    arm_length_cm: float = 20.0,
    grid_resolution: int = 160,
    orig_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Computes calibrated plant volume in cubic centimeters (cm^3) using the red trihedron
    reference frame to establish exact metric scale and coordinate axes.

    Parameters
    ----------
    mask_xy, mask_yz, mask_xz : 2D boolean numpy arrays
        Plant foliage masks from the top (XY) and side (YZ, XZ) views.
    kp_xy, kp_yz, kp_xz : TrihedronKeypoints
        Detected trihedron keypoints and metric scales for each view.
    arm_length_cm : float
        Physical length of each trihedron arm in centimeters (default 20.0 cm).
    grid_resolution : int
        Voxel grid subdivision along each axis (default 160 -> 4.09M voxels).
    orig_shape : tuple of (height, width), optional
        Original image dimensions used during keypoint detection (for scaling).

    Returns
    -------
    volume_cm3 : float
        Calibrated plant volume in cm^3.
    metadata : dict
        Detailed metrics including filled voxels, bounding box extents, and scales.
    """
    v_xy, s_xy_x, s_xy_y = _resolve_view_scale(kp_xy, mask_xy, orig_shape)
    v_xz, s_xz_x, s_xz_y = _resolve_view_scale(kp_xz, mask_xz, orig_shape)
    v_yz, s_yz_x, s_yz_y = _resolve_view_scale(kp_yz, mask_yz, orig_shape)

    # Locate plant bounding extents from 2D masks in physical cm
    ys_xz, xs_xz = np.where(mask_xz)
    ys_yz, xs_yz = np.where(mask_yz)
    ys_xy, xs_xy = np.where(mask_xy)

    if len(xs_xz) == 0 or len(xs_yz) == 0 or len(xs_xy) == 0:
        return 0.0, {"filled_voxels": 0, "error": "Empty plant mask in one or more views"}

    # X axis (cm from vertex):
    x_min_xz = (np.min(xs_xz) - v_xz[0]) / s_xz_x
    x_max_xz = (np.max(xs_xz) - v_xz[0]) / s_xz_x
    x_min_xy = (np.min(xs_xy) - v_xy[0]) / s_xy_x
    x_max_xy = (np.max(xs_xy) - v_xy[0]) / s_xy_x
    x_min = max(0.0, min(x_min_xz, x_min_xy) - 0.5)
    x_max = max(x_max_xz, x_max_xy) + 0.5

    # Y axis (cm from vertex):
    y_min_yz = (np.min(xs_yz) - v_yz[0]) / s_yz_x
    y_max_yz = (np.max(xs_yz) - v_yz[0]) / s_yz_x
    y_min_xy = (np.min(ys_xy) - v_xy[1]) / s_xy_y
    y_max_xy = (np.max(ys_xy) - v_xy[1]) / s_xy_y
    y_min = max(0.0, min(y_min_yz, y_min_xy) - 0.5)
    y_max = max(y_max_yz, y_max_xy) + 0.5

    # Z axis (cm above table surface Z=0):
    z_min_xz = (v_xz[1] - np.max(ys_xz)) / s_xz_y
    z_max_xz = (v_xz[1] - np.min(ys_xz)) / s_xz_y
    z_min_yz = (v_yz[1] - np.max(ys_yz)) / s_yz_y
    z_max_yz = (v_yz[1] - np.min(ys_yz)) / s_yz_y
    z_min = max(0.0, min(z_min_xz, z_min_yz) - 0.5)
    z_max = max(z_max_xz, z_max_yz) + 0.5

    # Voxel grid in physical space (cm)
    xs = np.linspace(x_min, x_max, grid_resolution)
    ys = np.linspace(y_min, y_max, grid_resolution)
    zs = np.linspace(z_min, z_max, grid_resolution)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    # Project to XZ view
    u_xz = np.round(v_xz[0] + X * s_xz_x).astype(int)
    v_xz_coords = np.round(v_xz[1] - Z * s_xz_y).astype(int)
    in_xz = np.zeros(X.shape, dtype=bool)
    val_xz = (u_xz >= 0) & (u_xz < mask_xz.shape[1]) & (v_xz_coords >= 0) & (v_xz_coords < mask_xz.shape[0])
    in_xz[val_xz] = mask_xz[v_xz_coords[val_xz], u_xz[val_xz]]

    # Project to YZ view
    u_yz = np.round(v_yz[0] + Y * s_yz_x).astype(int)
    v_yz_coords = np.round(v_yz[1] - Z * s_yz_y).astype(int)
    in_yz = np.zeros(Y.shape, dtype=bool)
    val_yz = (u_yz >= 0) & (u_yz < mask_yz.shape[1]) & (v_yz_coords >= 0) & (v_yz_coords < mask_yz.shape[0])
    in_yz[val_yz] = mask_yz[v_yz_coords[val_yz], u_yz[val_yz]]

    # Project to XY view
    u_xy = np.round(v_xy[0] + X * s_xy_x).astype(int)
    v_xy_coords = np.round(v_xy[1] + Y * s_xy_y).astype(int)
    in_xy = np.zeros(X.shape, dtype=bool)
    val_xy = (u_xy >= 0) & (u_xy < mask_xy.shape[1]) & (v_xy_coords >= 0) & (v_xy_coords < mask_xy.shape[0])
    in_xy[val_xy] = mask_xy[v_xy_coords[val_xy], u_xy[val_xy]]

    # Intersection of all views
    voxel_grid = in_xz & in_yz & in_xy
    filled_voxels = int(np.sum(voxel_grid))

    dx = (x_max - x_min) / grid_resolution
    dy = (y_max - y_min) / grid_resolution
    dz = (z_max - z_min) / grid_resolution
    voxel_vol_cm3 = dx * dy * dz
    volume_cm3 = float(filled_voxels * voxel_vol_cm3)

    metadata = {
        "filled_voxels": filled_voxels,
        "voxel_volume_cm3": float(voxel_vol_cm3),
        "x_range_cm": (float(x_min), float(x_max)),
        "y_range_cm": (float(y_min), float(y_max)),
        "z_range_cm": (float(z_min), float(z_max)),
        "scale_px_per_cm": {
            "xy": float(s_xy_x),
            "xz": float(s_xz_x),
            "yz": float(s_yz_x)
        },
        "arm_length_cm": float(arm_length_cm),
        "grid_resolution": int(grid_resolution)
    }

    return volume_cm3, metadata


def _align_side_projection(
    mask: np.ndarray,
    cm_per_px_u: float,
    cm_per_px_z: float,
) -> np.ndarray:
    """
    Align a side projection (XZ or YZ) per Chopin et al. (2017):
    the widest horizontal section is centered vertically at z=0 and horizontally at u=0.
    """
    rows = np.where(np.any(mask, axis=1))[0]
    widths = np.empty(len(rows), dtype=int)
    centers = np.empty(len(rows), dtype=float)
    for idx_r, r in enumerate(rows):
        cols = np.where(mask[r])[0]
        widths[idx_r] = int(cols[-1] - cols[0] + 1)
        centers[idx_r] = 0.5 * (cols[0] + cols[-1])

    max_w = int(np.max(widths))
    widest_idx = np.where(widths == max_w)[0]
    mid_idx = widest_idx[len(widest_idx) // 2]
    r_widest = float(rows[mid_idx])
    c_center = float(centers[mid_idx])

    rs, cs = np.where(mask)
    return np.column_stack([
        (cs - c_center) * cm_per_px_u,
        (r_widest - rs) * cm_per_px_z,
    ])


def _extract_quadrant_xy_arc(
    contour_xy: np.ndarray,
    points_xy: np.ndarray,
    sx: float,
    sy: float,
    tol_x: float,
    tol_y: float,
    n_segments: int,
    multi_component: bool,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract the arc-length-parameterized outer boundary curve of Gamma_xy in one quadrant
    from (x*, 0) to (0, y*), resampled to 2 * n_segments + 1 points.
    """
    qx = contour_xy[:, 0] * sx
    qy = contour_xy[:, 1] * sy
    m_in = (qx > 0.0) & (qy > 0.0)

    px_q = points_xy[:, 0] * sx
    py_q = points_xy[:, 1] * sy
    ax_x = px_q[(px_q >= 0.0) & (np.abs(py_q) <= tol_y)]
    ax_y = py_q[(py_q >= 0.0) & (np.abs(px_q) <= tol_x)]
    x_end = float(np.max(ax_x)) if len(ax_x) > 0 else 0.0
    y_end = float(np.max(ax_y)) if len(ax_y) > 0 else 0.0

    if not np.any(m_in) and (x_end <= 0.0 or y_end <= 0.0):
        return None

    pts_in = np.column_stack([qx[m_in], qy[m_in]])
    if len(pts_in) > 0:
        ang = np.arctan2(pts_in[:, 1], pts_in[:, 0])
        if multi_component:
            # Take outer radial envelope per angular bin to avoid zig-zags on disconnected pieces
            n_bins = max(64, n_segments)
            edges = np.linspace(0.0, np.pi / 2.0, n_bins + 1)
            bin_idx = np.clip(np.searchsorted(edges, ang, side="right") - 1, 0, n_bins - 1)
            radii = np.hypot(pts_in[:, 0], pts_in[:, 1])
            env_pts = []
            for b in range(n_bins):
                in_bin = bin_idx == b
                if np.any(in_bin):
                    best = np.argmax(radii[in_bin])
                    env_pts.append(pts_in[in_bin][best])
            pts_in = np.array(env_pts, dtype=float) if env_pts else np.empty((0, 2), dtype=float)
        else:
            pts_in = pts_in[np.argsort(ang)]

    pts = np.vstack([[x_end, 0.0], pts_in, [0.0, y_end]]) if len(pts_in) > 0 else np.array([[x_end, 0.0], [0.0, y_end]])

    diffs = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    pts = pts[np.concatenate([[True], diffs > 1e-9])]
    if len(pts) < 2:
        return None

    arc_s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))])
    if arc_s[-1] <= 0.0:
        return None

    s_unif = np.linspace(0.0, arc_s[-1], 2 * n_segments + 1)
    return np.interp(s_unif, arc_s, pts[:, 0]), np.interp(s_unif, arc_s, pts[:, 1])


def _side_height_profile(
    pts_side: np.ndarray,
    u_sign: float,
    z_sign: float,
    u_star: float,
    n_segments: int,
) -> np.ndarray:
    """Compute height H(u') along [0, u_star] from a side projection in one octant."""
    pu = pts_side[:, 0] * u_sign
    pz = pts_side[:, 1] * z_sign
    mask = (pu >= 0.0) & (pz >= 0.0)
    if not np.any(mask):
        return np.zeros(n_segments, dtype=float)

    pu = pu[mask]
    pz = pz[mask]
    u_uniq, inv = np.unique(np.round(pu, 6), return_inverse=True)
    z_max = np.zeros(len(u_uniq), dtype=float)
    np.maximum.at(z_max, inv, pz)

    u_grid = (np.arange(n_segments) + 0.5) * (u_star / n_segments)
    return np.interp(u_grid, u_uniq, z_max, left=z_max[0], right=0.0)


def projective_bounds_volume(
    mask_xy: np.ndarray,
    mask_yz: np.ndarray,
    mask_xz: np.ndarray,
    kp_xy: TrihedronKeypoints,
    kp_yz: TrihedronKeypoints,
    kp_xz: TrihedronKeypoints,
    arm_length_cm: float = 20.0,
    arm_width_cm: float = 0.5,
    n_segments: int = 100,
    orig_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Estimate plant volume in cubic centimeters (cm^3) from three orthogonal 2D
    projections using the upper/lower-bound method of Chopin, Laga, & Miklavcic (2017),
    with the red trihedron providing metric calibration.

    Upper bound (Loomis-Whitney):
        V_> = sqrt(A_xy * A_xz * A_yz)
    Lower bound:
        V_< = sum of 8 octant wedge integrals over aligned projections
    Estimated volume:
        V_UL = (V_< + V_>) / 2
    """
    if n_segments < 2:
        raise ValueError("n_segments must be at least 2")

    m_xy = np.asarray(mask_xy, dtype=bool)
    m_yz = np.asarray(mask_yz, dtype=bool)
    m_xz = np.asarray(mask_xz, dtype=bool)

    _, s_xy_x, s_xy_y = _resolve_view_scale(kp_xy, m_xy, orig_shape)
    _, s_xz_x, s_xz_z = _resolve_view_scale(kp_xz, m_xz, orig_shape)
    _, s_yz_y, s_yz_z = _resolve_view_scale(kp_yz, m_yz, orig_shape)

    cm_xy_x, cm_xy_y = 1.0 / s_xy_x, 1.0 / s_xy_y
    cm_xz_x, cm_xz_z = 1.0 / s_xz_x, 1.0 / s_xz_z
    cm_yz_y, cm_yz_z = 1.0 / s_yz_y, 1.0 / s_yz_z

    area_xy = float(np.sum(m_xy)) * cm_xy_x * cm_xy_y
    area_xz = float(np.sum(m_xz)) * cm_xz_x * cm_xz_z
    area_yz = float(np.sum(m_yz)) * cm_yz_y * cm_yz_z

    if area_xy <= 0.0 or area_xz <= 0.0 or area_yz <= 0.0:
        return 0.0, {
            "method": "chopin_projective_bounds",
            "volume_ul_cm3": 0.0,
            "lower_bound_cm3": 0.0,
            "upper_bound_cm3": 0.0,
            "areas_cm2": {"xy": area_xy, "xz": area_xz, "yz": area_yz},
            "error": "Empty plant mask in one or more views",
        }

    v_upper = float(np.sqrt(area_xy * area_xz * area_yz))

    # Align Gamma_xy so its centre of mass sits at the origin (0, 0)
    ys_xy, xs_xy = np.where(m_xy)
    xc_xy = float(np.mean(xs_xy))
    yc_xy = float(np.mean(ys_xy))

    m_xy_filled = binary_fill_holes(m_xy).astype(np.uint8)
    cnts_xy, _ = cv2.findContours(m_xy_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts_xy:
        return 0.0, {
            "method": "chopin_projective_bounds",
            "volume_ul_cm3": 0.0,
            "lower_bound_cm3": 0.0,
            "upper_bound_cm3": v_upper,
            "error": "No boundary contours found in XY mask",
        }

    cnt_xy = np.vstack([c.reshape(-1, 2) for c in cnts_xy]).astype(float)
    contour_xy = np.column_stack([(cnt_xy[:, 0] - xc_xy) * cm_xy_x, (cnt_xy[:, 1] - yc_xy) * cm_xy_y])
    f_ys, f_xs = np.where(m_xy_filled > 0)
    points_xy = np.column_stack([(f_xs - xc_xy) * cm_xy_x, (f_ys - yc_xy) * cm_xy_y])

    # Align side projections by their widest horizontal section
    pts_xz = _align_side_projection(m_xz, cm_xz_x, cm_xz_z)
    pts_yz = _align_side_projection(m_yz, cm_yz_y, cm_yz_z)

    v_lower = 0.0
    octant_volumes: List[float] = []
    tol_x = 0.6 * cm_xy_x
    tol_y = 0.6 * cm_xy_y
    multi_comp = len(cnts_xy) > 1

    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            bound = _extract_quadrant_xy_arc(
                contour_xy=contour_xy,
                points_xy=points_xy,
                sx=sx,
                sy=sy,
                tol_x=tol_x,
                tol_y=tol_y,
                n_segments=n_segments,
                multi_component=multi_comp,
            )
            if bound is None:
                octant_volumes.extend([0.0, 0.0])
                continue

            bx, by = bound
            mid = n_segments
            x_part_y = bx[mid + 1:]
            y_part_x = by[:mid][::-1]

            for sz in (1.0, -1.0):
                mx = (pts_xz[:, 0] * sx >= 0.0) & (pts_xz[:, 1] * sz >= 0.0)
                my = (pts_yz[:, 0] * sy >= 0.0) & (pts_yz[:, 1] * sz >= 0.0)
                if not np.any(mx) or not np.any(my):
                    octant_volumes.append(0.0)
                    continue

                x_star = float(np.max(pts_xz[mx, 0] * sx))
                y_star = float(np.max(pts_yz[my, 0] * sy))
                if x_star <= 0.0 or y_star <= 0.0:
                    octant_volumes.append(0.0)
                    continue

                h_x = _side_height_profile(pts_xz, sx, sz, x_star, n_segments)
                h_y = _side_height_profile(pts_yz, sy, sz, y_star, n_segments)

                dx = x_star / n_segments
                dy = y_star / n_segments
                v_oct = 0.5 * float(np.sum(h_x * y_part_x)) * dx + 0.5 * float(np.sum(h_y * x_part_y)) * dy
                v_lower += v_oct
                octant_volumes.append(float(v_oct))

    v_ul = 0.5 * (v_lower + v_upper)

    metadata: Dict[str, Any] = {
        "method": "chopin_projective_bounds",
        "volume_ul_cm3": float(v_ul),
        "lower_bound_cm3": float(v_lower),
        "upper_bound_cm3": float(v_upper),
        "areas_cm2": {
            "xy": float(area_xy),
            "xz": float(area_xz),
            "yz": float(area_yz),
        },
        "octant_volumes_cm3": octant_volumes,
        "scale_px_per_cm": {
            "xy": float(s_xy_x),
            "xz": float(s_xz_x),
            "yz": float(s_yz_y),
        },
        "arm_length_cm": float(arm_length_cm),
        "arm_width_cm": float(arm_width_cm),
        "n_segments": int(n_segments),
    }

    return float(v_ul), metadata
