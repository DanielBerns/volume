"""
volumen - Plant volume estimation using multi-view photography and reference frames.
"""
from .estimator import (
    ProjectiveBoundsVolumeStrategy,
    VisualHullVolumeStrategy,
    VolumeEstimationStrategy,
    estimate_volume,
    estimate_volume_with_projective_bounds,
    estimate_volume_with_trihedron,
)
from .reconstruction import perspective_visual_hull, projective_bounds_volume, trihedron_visual_hull
from .trihedron import (
    TrihedronKeypoints,
    draw_trihedron_overlay,
    extract_red_mask,
    find_trihedron_keypoints,
)

__all__ = [
    "VolumeEstimationStrategy",
    "VisualHullVolumeStrategy",
    "ProjectiveBoundsVolumeStrategy",
    "estimate_volume",
    "estimate_volume_with_trihedron",
    "estimate_volume_with_projective_bounds",
    "TrihedronKeypoints",
    "find_trihedron_keypoints",
    "extract_red_mask",
    "draw_trihedron_overlay",
    "perspective_visual_hull",
    "trihedron_visual_hull",
    "projective_bounds_volume",
    "main",
]

def main() -> None:
    print("volumen: plant volume estimation toolkit")
