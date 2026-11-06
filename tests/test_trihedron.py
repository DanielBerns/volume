from pathlib import Path

import cv2
import numpy as np
import pytest

from volumen.trihedron import (
    TrihedronKeypoints,
    draw_trihedron_overlay,
    extract_red_mask,
    find_trihedron_keypoints,
)


def test_extract_red_mask_synthetic():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Red in BGR is (0, 0, 255)
    img[50:100, 50:150] = (0, 0, 220)
    # Green in BGR is (0, 255, 0)
    img[120:170, 50:100] = (0, 220, 0)

    mask = extract_red_mask(img)
    assert mask[75, 100] == 255
    assert mask[140, 75] == 0
    assert mask[10, 10] == 0


def test_find_trihedron_keypoints_synthetic():
    # Create 600x600 image with an L-shape trihedron with realistic bar thickness (~16px)
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    # Origin at (100, 100), arms extending along X (to 400) and Y (down to 400)
    # Arm 1: horizontal bar along X (y=92..108, x=100..400)
    img[92:108, 100:400] = (20, 20, 240)
    # Arm 2: vertical bar along Y (y=100..400, x=92..108)
    img[100:400, 92:108] = (20, 20, 240)

    kp = find_trihedron_keypoints(img, view="xy", arm_length_cm=20.0)
    assert kp is not None
    assert isinstance(kp, TrihedronKeypoints)
    vx, vy = kp.vertex
    assert abs(vx - 100) < 20
    assert abs(vy - 100) < 20
    assert kp.arm_length_px > 200
    assert kp.px_per_cm > 10


def test_draw_trihedron_overlay():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    kp = TrihedronKeypoints(
        vertex=(50, 50),
        tip_x=(200, 50),
        tip_y=(50, 200),
        tip_z=(20, 20),
        arm_length_px=150.0,
        px_per_cm=7.5,
        reproj_error=1.0,
        rvec=np.zeros(3),
        tvec=np.zeros(3),
    )
    overlay = draw_trihedron_overlay(img, kp)
    assert overlay.shape == img.shape
    assert overlay.sum() > 0


def test_real_sample_if_present():
    sample_dir = Path.home() / "Info" / "volume" / "run_20260814_165805"
    if not (sample_dir / "photo_xy.jpg").exists():
        pytest.skip("Local sample run_20260814_165805 not found")

    for view in ["xy", "yz", "xz"]:
        img = cv2.imread(str(sample_dir / f"photo_{view}.jpg"))
        assert img is not None
        kp = find_trihedron_keypoints(img, view=view, arm_length_cm=20.0)
        assert kp is not None, f"Failed to detect trihedron in {view}"
        assert kp.arm_length_px > 400
        assert 20.0 < kp.px_per_cm < 50.0
