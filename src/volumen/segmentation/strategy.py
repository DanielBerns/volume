from abc import ABC, abstractmethod
import numpy as np
from PIL import Image

class SegmentationStrategy(ABC):
    @abstractmethod
    def create_mask(self, image_path: str, resolution: int, **kwargs) -> np.ndarray:
        """
        Creates a binary mask from an image.
        Returns a 2D boolean numpy array of shape (resolution, resolution).
        """
        pass

class HSVThresholdStrategy(SegmentationStrategy):
    def create_mask(self, image_path: str, resolution: int, is_contrasting_bg: bool = True, **kwargs) -> np.ndarray:
        img = Image.open(image_path).convert('HSV')
        img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
        hsv_array = np.array(img)

        # Allow passing thresholds via kwargs, fallback to defaults
        hue_min = kwargs.get('hue_min', 40)
        hue_max = kwargs.get('hue_max', 100)
        sat_min = kwargs.get('sat_min', 50)

        if is_contrasting_bg:
            hue_channel = hsv_array[:, :, 0]
            plant_mask = (hue_channel > hue_min) & (hue_channel < hue_max)
        else:
            saturation_channel = hsv_array[:, :, 1]
            plant_mask = saturation_channel > sat_min

        return plant_mask

class SampledColorStrategy(SegmentationStrategy):
    """
    A strategy that uses a set of sampled HSV pixels to define the mask,
    by selecting pixels that fall within a tolerance of the sampled values.
    """
    def create_mask(self, image_path: str, resolution: int, sampled_pixels: list = None, tolerance: int = 20, **kwargs) -> np.ndarray:
        img = Image.open(image_path).convert('HSV')
        # We need to compute mask at original resolution if sampled_pixels coordinates are original,
        # but the problem states sampled pixels are just colors, or we can just apply it after resize.
        # Assuming sampled_pixels is a list of [H, S, V] values.
        img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
        hsv_array = np.array(img)

        if not sampled_pixels:
            # Fallback to empty mask or default if no pixels provided
            return np.zeros((resolution, resolution), dtype=bool)

        mask = np.zeros((resolution, resolution), dtype=bool)
        samples = np.array(sampled_pixels) # shape (N, 3)
        
        # Simple color distance approach: for each pixel, find min distance to any sample
        # Since HSV is cylindrical, hue distance should be cyclic, but for simplicity
        # we can use absolute difference. Let's do a basic bound check.
        for h, s, v in samples:
            # Wrap around for hue (0-255 in Pillow HSV corresponds to 0-360 deg)
            hue_diff = np.abs(hsv_array[:, :, 0].astype(int) - int(h))
            hue_diff = np.minimum(hue_diff, 256 - hue_diff)
            
            sat_diff = np.abs(hsv_array[:, :, 1].astype(int) - int(s))
            # Ignore V for robustness against lighting variations, or include it with less weight
            
            # Condition: hue within tolerance, and saturation not too far
            match = (hue_diff < tolerance) & (sat_diff < tolerance * 2)
            mask = mask | match

        return mask
