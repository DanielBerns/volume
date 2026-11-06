"""
High-level volume estimation interface coordinating segmentation and reconstruction.
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .reconstruction import perspective_visual_hull, trihedron_visual_hull
from .trihedron import TrihedronKeypoints

__all__ = [
    "estimate_volume",
    "estimate_volume_with_trihedron",
]

def estimate_volume(
    mask_xy: np.ndarray,
    mask_yz: np.ndarray,
    mask_xz: np.ndarray,
    focal_length: float = 50.0,
    resolution: int = 256
) -> float:
    """
    Coordinates the legacy 3D volume estimation using uncalibrated perspective visual hull.
    Returns volume in m^3.
    """
    return perspective_visual_hull(
        mask_xy=mask_xy,
        mask_yz=mask_yz,
        mask_xz=mask_xz,
        focal_length=focal_length,
        resolution=resolution
    )

def estimate_volume_with_trihedron(
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
    Coordinates calibrated 3D plant volume estimation using the red trihedron
    as spatial and metric reference.

    Returns
    -------
    volume_cm3 : float
        Estimated plant volume in cubic centimeters (cm^3).
    metadata : dict
        Calibration and reconstruction diagnostic metadata.
    """
    return trihedron_visual_hull(
        mask_xy=mask_xy,
        mask_yz=mask_yz,
        mask_xz=mask_xz,
        kp_xy=kp_xy,
        kp_yz=kp_yz,
        kp_xz=kp_xz,
        arm_length_cm=arm_length_cm,
        grid_resolution=grid_resolution,
        orig_shape=orig_shape,
    )
