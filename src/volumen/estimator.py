"""
High-level volume estimation interface coordinating segmentation and reconstruction.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .reconstruction import perspective_visual_hull, projective_bounds_volume, trihedron_visual_hull
from .trihedron import TrihedronKeypoints

__all__ = [
    "VolumeEstimationStrategy",
    "VisualHullVolumeStrategy",
    "ProjectiveBoundsVolumeStrategy",
    "estimate_volume",
    "estimate_volume_with_trihedron",
    "estimate_volume_with_projective_bounds",
]


class VolumeEstimationStrategy(ABC):
    """Strategy interface for calibrated 3D plant volume estimation from orthogonal masks."""

    @abstractmethod
    def estimate(
        self,
        mask_xy: np.ndarray,
        mask_yz: np.ndarray,
        mask_xz: np.ndarray,
        kp_xy: TrihedronKeypoints,
        kp_yz: TrihedronKeypoints,
        kp_xz: TrihedronKeypoints,
        arm_length_cm: float = 20.0,
        arm_width_cm: float = 0.5,
        orig_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Return (volume_cm3, metadata) given 3 orthogonal masks and trihedron keypoints."""


class VisualHullVolumeStrategy(VolumeEstimationStrategy):
    """Calibrated 3D voxel-carving visual hull volume estimation."""

    def __init__(self, grid_resolution: int = 160) -> None:
        self.grid_resolution = grid_resolution

    def estimate(
        self,
        mask_xy: np.ndarray,
        mask_yz: np.ndarray,
        mask_xz: np.ndarray,
        kp_xy: TrihedronKeypoints,
        kp_yz: TrihedronKeypoints,
        kp_xz: TrihedronKeypoints,
        arm_length_cm: float = 20.0,
        arm_width_cm: float = 0.5,
        orig_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        return trihedron_visual_hull(
            mask_xy=mask_xy,
            mask_yz=mask_yz,
            mask_xz=mask_xz,
            kp_xy=kp_xy,
            kp_yz=kp_yz,
            kp_xz=kp_xz,
            arm_length_cm=arm_length_cm,
            grid_resolution=self.grid_resolution,
            orig_shape=orig_shape,
        )


class ProjectiveBoundsVolumeStrategy(VolumeEstimationStrategy):
    """
    Calibrated volume estimation using the upper/lower-bound method from three 2D
    orthogonal projective images (Chopin, Laga, & Miklavcic, 2017).
    """

    def __init__(self, n_segments: int = 100) -> None:
        self.n_segments = n_segments

    def estimate(
        self,
        mask_xy: np.ndarray,
        mask_yz: np.ndarray,
        mask_xz: np.ndarray,
        kp_xy: TrihedronKeypoints,
        kp_yz: TrihedronKeypoints,
        kp_xz: TrihedronKeypoints,
        arm_length_cm: float = 20.0,
        arm_width_cm: float = 0.5,
        orig_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        return projective_bounds_volume(
            mask_xy=mask_xy,
            mask_yz=mask_yz,
            mask_xz=mask_xz,
            kp_xy=kp_xy,
            kp_yz=kp_yz,
            kp_xz=kp_xz,
            arm_length_cm=arm_length_cm,
            arm_width_cm=arm_width_cm,
            n_segments=self.n_segments,
            orig_shape=orig_shape,
        )


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
    orig_shape: Optional[Tuple[int, int]] = None,
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
    return VisualHullVolumeStrategy(grid_resolution=grid_resolution).estimate(
        mask_xy=mask_xy,
        mask_yz=mask_yz,
        mask_xz=mask_xz,
        kp_xy=kp_xy,
        kp_yz=kp_yz,
        kp_xz=kp_xz,
        arm_length_cm=arm_length_cm,
        orig_shape=orig_shape,
    )


def estimate_volume_with_projective_bounds(
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
    Coordinates calibrated plant volume estimation using the Chopin et al. (2017)
    upper/lower-bound method and the red trihedron (default 200 mm length, 5 mm width)
    as metric reference.

    Returns
    -------
    volume_cm3 : float
        Estimated plant volume V_UL = (V_< + V_>) / 2 in cm^3.
    metadata : dict
        Lower bound, upper bound, projection areas, octant volumes, and scale metadata.
    """
    return ProjectiveBoundsVolumeStrategy(n_segments=n_segments).estimate(
        mask_xy=mask_xy,
        mask_yz=mask_yz,
        mask_xz=mask_xz,
        kp_xy=kp_xy,
        kp_yz=kp_yz,
        kp_xz=kp_xz,
        arm_length_cm=arm_length_cm,
        arm_width_cm=arm_width_cm,
        orig_shape=orig_shape,
    )
