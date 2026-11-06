"""
3D visual hull reconstruction algorithms, supporting perspective and trihedron-calibrated reference frames.
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np

from volumen.trihedron import TrihedronKeypoints

__all__ = [
    "compute_intrinsic_matrix",
    "compute_extrinsic_matrix",
    "perspective_visual_hull",
    "trihedron_visual_hull",
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

def trihedron_visual_hull(
    mask_xy: np.ndarray,
    mask_yz: np.ndarray,
    mask_xz: np.ndarray,
    kp_xy: TrihedronKeypoints,
    kp_yz: TrihedronKeypoints,
    kp_xz: TrihedronKeypoints,
    arm_length_cm: float = 20.0,
    grid_resolution: int = 160,
    orig_shape: Optional[Tuple[int, int]] = (1496, 1496),
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
    orig_shape : tuple of (height, width)
        Original image dimensions used during keypoint detection (for scaling).

    Returns
    -------
    volume_cm3 : float
        Calibrated plant volume in cm^3.
    metadata : dict
        Detailed metrics including filled voxels, bounding box extents, and scales.
    """
    # Compute resolution scale factors if masks were resized
    def get_scaled_keypoints(kp: TrihedronKeypoints, mask: np.ndarray):
        h_m, w_m = mask.shape[:2]
        if orig_shape is not None:
            sx = w_m / float(orig_shape[1])
            sy = h_m / float(orig_shape[0])
        else:
            sx, sy = 1.0, 1.0
        v_scaled = (kp.vertex[0] * sx, kp.vertex[1] * sy)
        s_scaled = kp.px_per_cm * sx
        return v_scaled, s_scaled

    v_xy, s_xy = get_scaled_keypoints(kp_xy, mask_xy)
    v_xz, s_xz = get_scaled_keypoints(kp_xz, mask_xz)
    v_yz, s_yz = get_scaled_keypoints(kp_yz, mask_yz)

    # Locate plant bounding extents from 2D masks in physical cm
    ys_xz, xs_xz = np.where(mask_xz)
    ys_yz, xs_yz = np.where(mask_yz)
    ys_xy, xs_xy = np.where(mask_xy)

    if len(xs_xz) == 0 or len(xs_yz) == 0 or len(xs_xy) == 0:
        return 0.0, {"filled_voxels": 0, "error": "Empty plant mask in one or more views"}

    # X axis (cm from vertex):
    x_min_xz = (np.min(xs_xz) - v_xz[0]) / s_xz
    x_max_xz = (np.max(xs_xz) - v_xz[0]) / s_xz
    x_min_xy = (np.min(xs_xy) - v_xy[0]) / s_xy
    x_max_xy = (np.max(xs_xy) - v_xy[0]) / s_xy
    x_min = max(0.0, min(x_min_xz, x_min_xy) - 0.5)
    x_max = max(x_max_xz, x_max_xy) + 0.5

    # Y axis (cm from vertex):
    y_min_yz = (np.min(xs_yz) - v_yz[0]) / s_yz
    y_max_yz = (np.max(xs_yz) - v_yz[0]) / s_yz
    y_min_xy = (np.min(ys_xy) - v_xy[1]) / s_xy
    y_max_xy = (np.max(ys_xy) - v_xy[1]) / s_xy
    y_min = max(0.0, min(y_min_yz, y_min_xy) - 0.5)
    y_max = max(y_max_yz, y_max_xy) + 0.5

    # Z axis (cm above table surface Z=0):
    z_min_xz = (v_xz[1] - np.max(ys_xz)) / s_xz
    z_max_xz = (v_xz[1] - np.min(ys_xz)) / s_xz
    z_min_yz = (v_yz[1] - np.max(ys_yz)) / s_yz
    z_max_yz = (v_yz[1] - np.min(ys_yz)) / s_yz
    z_min = max(0.0, min(z_min_xz, z_min_yz) - 0.5)
    z_max = max(z_max_xz, z_max_yz) + 0.5

    # Voxel grid in physical space (cm)
    xs = np.linspace(x_min, x_max, grid_resolution)
    ys = np.linspace(y_min, y_max, grid_resolution)
    zs = np.linspace(z_min, z_max, grid_resolution)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    # Project to XZ view
    u_xz = np.round(v_xz[0] + X * s_xz).astype(int)
    v_xz_coords = np.round(v_xz[1] - Z * s_xz).astype(int)
    in_xz = np.zeros(X.shape, dtype=bool)
    val_xz = (u_xz >= 0) & (u_xz < mask_xz.shape[1]) & (v_xz_coords >= 0) & (v_xz_coords < mask_xz.shape[0])
    in_xz[val_xz] = mask_xz[v_xz_coords[val_xz], u_xz[val_xz]]

    # Project to YZ view
    u_yz = np.round(v_yz[0] + Y * s_yz).astype(int)
    v_yz_coords = np.round(v_yz[1] - Z * s_yz).astype(int)
    in_yz = np.zeros(Y.shape, dtype=bool)
    val_yz = (u_yz >= 0) & (u_yz < mask_yz.shape[1]) & (v_yz_coords >= 0) & (v_yz_coords < mask_yz.shape[0])
    in_yz[val_yz] = mask_yz[v_yz_coords[val_yz], u_yz[val_yz]]

    # Project to XY view
    u_xy = np.round(v_xy[0] + X * s_xy).astype(int)
    v_xy_coords = np.round(v_xy[1] + Y * s_xy).astype(int)
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
            "xy": float(s_xy),
            "xz": float(s_xz),
            "yz": float(s_yz)
        },
        "arm_length_cm": float(arm_length_cm),
        "grid_resolution": int(grid_resolution)
    }

    return volume_cm3, metadata
