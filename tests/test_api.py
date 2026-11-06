import io

import cv2
import numpy as np
from fastapi.testclient import TestClient

from website.main import app

client = TestClient(app)


def test_index_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Plant Volume Estimator" in response.text
    assert "Red Trihedron Reference" in response.text
    assert "cm³" in response.text


def test_segment_api_endpoint(tmp_path):
    # Create simple synthetic test images (300x300) with a red trihedron and green plant
    def make_img(view):
        img = np.full((300, 300, 3), 120, dtype=np.uint8)
        # Red bar (arm 1)
        img[90:110, 100:280] = (20, 20, 240)
        # Red bar (arm 2)
        if view == "xy":
            img[100:280, 90:110] = (20, 20, 240)
        else:
            img[20:100, 90:110] = (20, 20, 240)
        # Green plant foliage
        img[120:160, 120:160] = (30, 180, 40)
        _, buf = cv2.imencode(".jpg", img)
        return io.BytesIO(buf.tobytes())

    files = {
        "photo_xy": ("photo_xy.jpg", make_img("xy"), "image/jpeg"),
        "photo_yz": ("photo_yz.jpg", make_img("yz"), "image/jpeg"),
        "photo_xz": ("photo_xz.jpg", make_img("xz"), "image/jpeg"),
    }
    data = {
        "strategy": "trihedron",
        "arm_length_cm": 20.0,
    }

    res = client.post("/api/segment", files=files, data=data)
    assert res.status_code == 200
    res_data = res.json()
    assert "run_id" in res_data
    assert "estimated_volume_cm3" in res_data
    assert "estimated_volume_m3" in res_data
    assert "masks" in res_data
