import numpy as np

from volumen.estimator import (
    ProjectiveBoundsVolumeStrategy,
    estimate_volume_with_projective_bounds,
    estimate_volume_with_trihedron,
)
from volumen.reconstruction import projective_bounds_volume, trihedron_visual_hull
from volumen.trihedron import TrihedronKeypoints


def _make_dummy_kp(v, px_per_cm=10.0, image_shape=None):
    return TrihedronKeypoints(
        vertex=v,
        tip_x=(v[0] + 20.0 * px_per_cm, v[1]),
        tip_y=(v[0], v[1] + 20.0 * px_per_cm),
        tip_z=(v[0], v[1] - 20.0 * px_per_cm),
        arm_length_px=20.0 * px_per_cm,
        px_per_cm=px_per_cm,
        reproj_error=1.0,
        rvec=np.zeros(3),
        tvec=np.zeros(3),
        image_shape=image_shape,
    )


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

    kp_xy = _make_dummy_kp((50, 50))
    kp_yz = _make_dummy_kp((50, 300))
    kp_xz = _make_dummy_kp((50, 300))

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

    kp_xy = _make_dummy_kp((50, 50))
    kp_yz = _make_dummy_kp((50, 300))
    kp_xz = _make_dummy_kp((50, 300))

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


def _rasterize_orthogonal_masks(occ_fn, n=200, px_per_cm=10.0):
    coords = (np.arange(n) - n / 2.0) / px_per_cm
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    occ = occ_fn(x, y, z)
    m_xy = np.any(occ, axis=2).T
    m_xz = np.any(occ, axis=1).T[::-1]
    m_yz = np.any(occ, axis=0).T[::-1]
    return m_xy, m_yz, m_xz


def test_projective_bounds_sphere():
    radius_cm = 4.38
    true_vol = (4.0 / 3.0) * np.pi * (radius_cm ** 3)  # ~351.97 cm^3
    m_xy, m_yz, m_xz = _rasterize_orthogonal_masks(
        lambda x, y, z: (x**2 + y**2 + z**2) <= radius_cm**2,
        n=180,
        px_per_cm=10.0,
    )
    kp = _make_dummy_kp((20, 20), px_per_cm=10.0, image_shape=(180, 180))

    vol_cm3, meta = estimate_volume_with_projective_bounds(
        mask_xy=m_xy,
        mask_yz=m_yz,
        mask_xz=m_xz,
        kp_xy=kp,
        kp_yz=kp,
        kp_xz=kp,
        arm_length_cm=20.0,
        arm_width_cm=0.5,
    )

    assert meta["method"] == "chopin_projective_bounds"
    assert meta["lower_bound_cm3"] < true_vol < meta["upper_bound_cm3"]
    assert abs(vol_cm3 - true_vol) / true_vol < 0.06
    assert abs(sum(meta["octant_volumes_cm3"]) - meta["lower_bound_cm3"]) < 1e-6


def test_projective_bounds_ellipsoid_and_cone():
    # Ellipsoid (high curvature): V_UL should be accurate (< 6% error)
    a, b, c = 3.0, 4.0, 3.5
    true_ell = (4.0 / 3.0) * np.pi * a * b * c
    m_xy, m_yz, m_xz = _rasterize_orthogonal_masks(
        lambda x, y, z: (x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2 <= 1.0,
        n=160,
        px_per_cm=10.0,
    )
    kp = _make_dummy_kp((20, 20), px_per_cm=10.0, image_shape=(160, 160))
    vol_ell, meta_ell = ProjectiveBoundsVolumeStrategy(n_segments=100).estimate(
        m_xy, m_yz, m_xz, kp, kp, kp, arm_length_cm=20.0, arm_width_cm=0.5
    )
    assert meta_ell["lower_bound_cm3"] < true_ell < meta_ell["upper_bound_cm3"]
    assert abs(vol_ell - true_ell) / true_ell < 0.06

    # Upright cone resting on z=0 (4 lower octants have zero height)
    r_cone, h_cone = 4.0, 6.0
    true_cone = (1.0 / 3.0) * np.pi * (r_cone ** 2) * h_cone
    m_xy_c, m_yz_c, m_xz_c = _rasterize_orthogonal_masks(
        lambda x, y, z: (np.sqrt(x**2 + y**2) / r_cone + z / h_cone <= 1.0) & (z >= 0.0),
        n=160,
        px_per_cm=10.0,
    )
    vol_cone, meta_cone = projective_bounds_volume(
        m_xy_c, m_yz_c, m_xz_c, kp, kp, kp, arm_length_cm=20.0, arm_width_cm=0.5
    )
    assert meta_cone["lower_bound_cm3"] <= true_cone <= meta_cone["upper_bound_cm3"]
    assert vol_cone > 0.0
    # Lower half octants should be ~0 while upper half octants are positive
    assert sum(meta_cone["octant_volumes_cm3"]) > 0.85 * true_cone


def test_projective_bounds_cube_and_non_square_scaling():
    # 8 cm cube -> true = 512 cm^3; V_upper ~ 512, V_lower ~ 256, V_UL ~ 384
    m_xy = np.zeros((200, 200), dtype=bool)
    m_xz = np.zeros((200, 200), dtype=bool)
    m_yz = np.zeros((200, 200), dtype=bool)
    m_xy[60:140, 60:140] = True
    m_xz[60:140, 60:140] = True
    m_yz[60:140, 60:140] = True

    # Simulate keypoints detected on a 400x300 (non-square) photo scaled to 200x200 mask
    kp = _make_dummy_kp((20, 20), px_per_cm=10.0, image_shape=(300, 400))
    # Each 80x80 mask square represents 16 cm (width) x 12 cm (height) = 192 cm^2
    # Upper bound = sqrt(192^3) = 2660.43 cm^3; cube lower bound = 0.5 * upper; V_UL = 0.75 * upper
    expected_upper = (16.0 * 12.0) ** 1.5
    vol_box, meta_box = projective_bounds_volume(
        m_xy, m_yz, m_xz, kp, kp, kp, arm_length_cm=20.0, arm_width_cm=0.5
    )
    assert abs(meta_box["upper_bound_cm3"] - expected_upper) < 30.0
    assert abs(meta_box["lower_bound_cm3"] - 0.5 * expected_upper) < 80.0
    assert abs(vol_box - 0.75 * expected_upper) < 50.0


def test_projective_bounds_multicomponent_and_empty_mask():
    m_xy = np.zeros((128, 128), dtype=bool)
    m_xz = np.zeros((128, 128), dtype=bool)
    m_yz = np.zeros((128, 128), dtype=bool)

    # Disconnected foliage clusters
    m_xy[40:80, 40:80] = True
    m_xy[30:40, 85:95] = True
    m_xz[30:90, 40:85] = True
    m_yz[30:90, 40:85] = True
    kp = _make_dummy_kp((10, 10), px_per_cm=10.0, image_shape=(128, 128))

    vol_multi, meta_multi = projective_bounds_volume(m_xy, m_yz, m_xz, kp, kp, kp)
    assert vol_multi > 0.0
    assert meta_multi["lower_bound_cm3"] > 0.0
    assert meta_multi["upper_bound_cm3"] > 0.0

    # Empty mask returns 0.0 cleanly
    empty = np.zeros((128, 128), dtype=bool)
    vol_empty, meta_empty = projective_bounds_volume(empty, m_yz, m_xz, kp, kp, kp)
    assert vol_empty == 0.0
    assert "error" in meta_empty
