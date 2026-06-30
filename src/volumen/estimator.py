from typing import Optional
import numpy as np

from .reconstruction import perspective_visual_hull

def estimate_volume(
    mask_xy: np.ndarray, 
    mask_yz: np.ndarray, 
    mask_xz: np.ndarray,
    focal_length: float = 50.0,
    resolution: int = 256
) -> float:
    """
    Coordinates the 3D volume estimation using perspective visual hull.
    """
    return perspective_visual_hull(
        mask_xy=mask_xy,
        mask_yz=mask_yz,
        mask_xz=mask_xz,
        focal_length=focal_length,
        resolution=resolution
    )
