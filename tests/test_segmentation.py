from pathlib import Path

import cv2
import numpy as np

from volumen.segmentation.strategy import TrihedronPlantSegmentationStrategy


def test_trihedron_plant_segmentation(tmp_path: Path):
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Background: gray tabletop (BGR 100, 100, 100)
    img[:] = (100, 100, 100)

    # Red trihedron bar: BGR (20, 20, 220)
    img[20:40, 20:250] = (20, 20, 220)

    # Plant foliage: Green needles, BGR (30, 160, 40)
    img[100:180, 100:180] = (30, 160, 40)

    img_file = tmp_path / "test_photo.jpg"
    cv2.imwrite(str(img_file), img)

    strategy = TrihedronPlantSegmentationStrategy()
    mask = strategy.create_mask(str(img_file), resolution=256)

    assert mask.shape == (256, 256)
    assert mask.dtype == bool

    # The plant foliage area should be marked True
    # The red bar area should be subtracted (False)
    # The gray tabletop should be background (False)
    assert mask.sum() > 0
    # Center of plant
    plant_center_res = int(140 * 256 / 300)
    assert mask[plant_center_res, plant_center_res]

    # Red bar area
    red_bar_res = int(30 * 256 / 300)
    assert not mask[red_bar_res, plant_center_res]
