import numpy as np

from volumen.estimator import estimate_volume_with_trihedron
from volumen.reconstruction import trihedron_visual_hull
from volumen.trihedron import TrihedronKeypoints


def test_trihedron_visual_hull_cube():
    # Construct a synthetic 3D cube of 10 cm x 10 cm x 10 cm = 1000 cm3
    # Arm length L = 20 cm, scale = 10 px/cm => arm_length_px = 200 px
    # Cube occupies [2, 12] cm in X, Y, Z.
    H, W = 400, 400
    mask_xy = np.zeros((H, W), dtype=bool)
    mask_yz = np.zeros((H, W), dtype=bool)
    mask_xz = np.zeros((H, W), dtype=bool)

    # XY view: vertex at (50, 50). X rightwards, Y downwards.
    # X in [2, 12] cm -> u in [70, 170]
    # Y in [2, 12] cm -> v in [70, 170]
    mask_xy[70:170, 70:170] = True

    # XZ view: vertex at (50, 300). Table surface at y=300.
    # X in [2, 12] cm -> u in [70, 170]
    # Z in [2, 12] cm -> v in 300 - [120, 20] = [180, 280]
    mask_xz[180:280, 70:170] = True

    # YZ view: vertex at (50, 300). Table surface at y=300.
    # Y in [2, 12] cm -> u in [70, 170]
    # Z in [2, 12] cm -> v in 300 - [120, 20] = [180, 280]
    mask_yz[180:280, 70:170] = True

    def dummy_kp(v):
        return TrihedronKeypoints(
            vertex=v,
            tip_x=(v[0] + 200, v[1]),
            tip_y=(v[0], v[1] + 200),
            tip_z=(v[0], v[1] - 200),
            arm_length_px=200.0,
            px_per_cm=10.0,
            reproj_error=1.0,
            rvec=np.zeros(3),
            tvec=np.zeros(3),
        )

    kp_xy = dummy_kp((50, 50))
    kp_yz = dummy_kp((50, 300))
    kp_xz = dummy_kp((50, 300))

    vol_cm3, meta = trihedron_visual_hull(
        mask_xy=mask_xy,
        mask_yz=mask_yz,
        mask_xz=mask_xz,
        kp_xy=kp_xy,
        kp_yz=kp_yz,
        kp_xz=kp_xz,
        arm_length_cm=20.0,
        grid_resolution=80,
        orig_shape=(H, W),
    )

    # True volume is 10 x 10 x 10 = 1000 cm3
    # Allow boundary discretization error (< 10%)
    assert abs(vol_cm3 - 1000.0) < 100.0
    assert meta["filled_voxels"] > 0


def test_estimate_volume_with_trihedron():
    H, W = 400, 400
    mask_xy = np.zeros((H, W), dtype=bool)
    mask_yz = np.zeros((H, W), dtype=bool)
    mask_xz = np.zeros((H, W), dtype=bool)

    # 6 cm cube: [2, 8] cm
    mask_xy[70:130, 70:130] = True
    mask_xz[220:280, 70:130] = True
    mask_yz[220:280, 70:130] = True

    def dummy_kp(v):
        return TrihedronKeypoints(
            vertex=v,
            tip_x=(v[0] + 200, v[1]),
            tip_y=(v[0], v[1] + 200),
            tip_z=(v[0], v[1] - 200),
            arm_length_px=200.0,
            px_per_cm=10.0,
            reproj_error=1.0,
            rvec=np.zeros(3),
            tvec=np.zeros(3),
        )

    kp_xy = dummy_kp((50, 50))
    kp_yz = dummy_kp((50, 300))
    kp_xz = dummy_kp((50, 300))

    vol_cm3, meta = estimate_volume_with_trihedron(
        mask_xy=mask_xy,
        mask_yz=mask_yz,
        mask_xz=mask_xz,
        kp_xy=kp_xy,
        kp_yz=kp_yz,
        kp_xz=kp_xz,
        arm_length_cm=20.0,
        grid_resolution=60,
        orig_shape=(H, W),
    )

    # Bounding box is 6 cm x 6 cm x 6 cm = 216 cm3
    assert abs(vol_cm3 - 216.0) < 40.0
    assert meta["filled_voxels"] > 0
