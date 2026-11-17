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
    assert "Direct-Method Volume" in response.text
    assert "Correlation Study: Direct vs. Estimated Volume" in response.text
    assert 'id="workspaceDirectForm"' in response.text
    assert 'id="correlationDirectForm"' in response.text


def _make_img(view: str) -> io.BytesIO:
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


def test_segment_api_endpoint(tmp_path, monkeypatch):
    import website.main as web_main

    monkeypatch.setattr(web_main, "VOLUME_DIR", tmp_path)

    files = {
        "photo_xy": ("photo_xy.jpg", _make_img("xy"), "image/jpeg"),
        "photo_yz": ("photo_yz.jpg", _make_img("yz"), "image/jpeg"),
        "photo_xz": ("photo_xz.jpg", _make_img("xz"), "image/jpeg"),
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
    assert res_data["direct_volume_cm3"] is None


def test_direct_volume_and_correlation_workflow(tmp_path, monkeypatch):
    import json

    import website.main as web_main

    monkeypatch.setattr(web_main, "VOLUME_DIR", tmp_path)

    # Create three synthetic runs with known estimates and direct measures
    samples = [
        ("run_20261117_100001", "Sample A", 10.0, 12.0),
        ("run_20261117_100002", "Sample B", 20.0, 22.0),
        ("run_20261117_100003", "Sample C", 30.0, 31.0),
    ]
    for run_id, name, direct, est in samples:
        run_dir = tmp_path / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "measurement.json").write_text(
            json.dumps({
                "run_id": run_id,
                "sample_name": name,
                "strategy": "trihedron",
                "estimated_volume_cm3": est,
                "estimated_volume_m3": est / 1e6,
                "direct_volume_cm3": None,
                "direct_method": None,
            }),
            encoding="utf-8",
        )

    # Save direct measurement for first two samples via API
    r1 = client.post("/api/direct-volume", json={
        "run_id": "run_20261117_100001",
        "direct_volume_cm3": 10.0,
        "sample_name": "Sample A",
        "direct_method": "Water displacement (Archimedes)",
    })
    assert r1.status_code == 200
    assert r1.json()["correlation"]["stats"]["n"] == 1
    assert r1.json()["correlation"]["stats"]["pearson_r"] is None

    r2 = client.post("/api/direct-volume", json={
        "run_id": "run_20261117_100002",
        "direct_volume_cm3": 20.0,
        "direct_method": "Graduated cylinder",
    })
    assert r2.status_code == 200
    assert r2.json()["correlation"]["stats"]["n"] == 2
    assert r2.json()["correlation"]["stats"]["pearson_r"] == 1.0

    r3 = client.post("/api/direct-volume", json={
        "run_id": "run_20261117_100003",
        "direct_volume_cm3": 30.0,
    })
    assert r3.status_code == 200
    corr = r3.json()["correlation"]
    stats = corr["stats"]
    assert stats["n"] == 3
    assert stats["pearson_r"] > 0.99
    assert stats["r_squared"] > 0.99
    assert stats["spearman_rho"] == 1.0
    assert stats["slope"] == 0.95
    assert stats["intercept"] == 2.6667
    assert stats["mae_cm3"] == 1.67
    assert stats["mean_bias_cm3"] == 1.67

    # GET /api/correlation returns the same summary
    get_res = client.get("/api/correlation")
    assert get_res.status_code == 200
    assert get_res.json()["stats"]["n"] == 3

    # Export CSV works and contains headers + rows
    csv_res = client.get("/api/correlation/export")
    assert csv_res.status_code == 200
    assert "text/csv" in csv_res.headers["content-type"]
    lines = csv_res.text.strip().splitlines()
    assert len(lines) == 4
    assert "direct_volume_cm3,estimated_volume_cm3" in lines[0]

    # Delete direct measurement clears it without deleting the run
    del_res = client.delete("/api/direct-volume/run_20261117_100002")
    assert del_res.status_code == 200
    assert del_res.json()["correlation"]["stats"]["n"] == 2
    assert len(del_res.json()["correlation"]["runs"]) == 3


def test_direct_volume_validation_and_segment_optional_fields(tmp_path, monkeypatch):
    import website.main as web_main

    monkeypatch.setattr(web_main, "VOLUME_DIR", tmp_path)

    files = {
        "photo_xy": ("photo_xy.jpg", _make_img("xy"), "image/jpeg"),
        "photo_yz": ("photo_yz.jpg", _make_img("yz"), "image/jpeg"),
        "photo_xz": ("photo_xz.jpg", _make_img("xz"), "image/jpeg"),
    }
    data = {
        "strategy": "hsv",
        "direct_volume_cm3": "45.5",
        "sample_name": "Plant 01",
        "direct_method": "Water displacement",
    }
    res = client.post("/api/segment", files=files, data=data)
    assert res.status_code == 200
    body = res.json()
    assert body["direct_volume_cm3"] == 45.5
    assert body["sample_name"] == "Plant 01"
    assert body["direct_method"] == "Water displacement"

    # Reject non-positive direct volume
    bad_val = client.post("/api/direct-volume", json={
        "run_id": body["run_id"],
        "direct_volume_cm3": -5.0,
    })
    assert bad_val.status_code == 422

    # Reject invalid/path-traversal run_id
    bad_id = client.post("/api/direct-volume", json={
        "run_id": "../etc",
        "direct_volume_cm3": 10.0,
    })
    assert bad_id.status_code == 400

