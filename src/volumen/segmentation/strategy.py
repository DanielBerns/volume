from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import cv2
from volumen.svdcodes import (
    get_pixels,
    get_transformed_data,
    get_codes,
    get_segments,
    get_segments_mosaic_view,
)

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

class GrabCutStrategy(SegmentationStrategy):
    """
    A strategy that uses OpenCV's GrabCut algorithm for automatic foreground extraction.
    """
    def create_mask(self, image_path: str, resolution: int, **kwargs) -> np.ndarray:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
        cv_img = np.array(img)
        
        # Convert RGB to BGR for OpenCV
        cv_img = cv_img[:, :, ::-1]
        
        mask = np.zeros(cv_img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        fg_rects = kwargs.get('fg_rects', [])
        bg_rects = kwargs.get('bg_rects', [])
        
        # Define a bounding box slightly smaller than the image
        margin = int(resolution * 0.05)
        rect = (margin, margin, resolution - 2*margin, resolution - 2*margin)
        
        try:
            if not fg_rects and not bg_rects:
                cv2.grabCut(cv_img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            else:
                # Initialize mask with GC_BGD everywhere
                mask[:] = cv2.GC_BGD
                # Make the central region PR_BGD or PR_FGD
                mask[margin:resolution-margin, margin:resolution-margin] = cv2.GC_PR_BGD
                
                # We need to make sure there's at least some PR_FGD or FGD
                if not fg_rects:
                     mask[margin*2:resolution-margin*2, margin*2:resolution-margin*2] = cv2.GC_PR_FGD
                     
                for (x, y, w, h) in bg_rects:
                    mask[y:y+h, x:x+w] = cv2.GC_BGD
                for (x, y, w, h) in fg_rects:
                    mask[y:y+h, x:x+w] = cv2.GC_FGD
                    
                cv2.grabCut(cv_img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        except Exception as e:
            print(f"GrabCut failed: {e}")
            return np.zeros((resolution, resolution), dtype=bool)
            
        # 0 and 2 are background, 1 and 3 are foreground
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        return mask2.astype(bool)


class SVDCodesStrategy(SegmentationStrategy):
    """
    A strategy that applies the SVDCodes algorithm for image segmentation.

    The algorithm works by:
      1. Flattening the RGB channels into a pixel matrix.
      2. Computing the top-2 principal components via SVD.
      3. Encoding each pixel as an integer code from those components.
      4. Grouping pixels that share the same code and colouring each group
         with the representative (centroid-nearest) pixel of that group.

    ``create_mask`` returns a boolean mask where ``True`` marks pixels that
    belong to non-background (non-black) segments.

    Accepted ``kwargs``::

        alpha (int): Controls the quantisation granularity of the codes.
            Higher values produce more segments.  Defaults to 16.
        view_type (str): Either ``"segments"`` (default) for the flat
            segmented image, or ``"mosaic view"`` for a multi-panel layout
            of the largest segments stacked vertically.
    """

    def create_mask(
        self,
        image_path: str,
        resolution: int,
        **kwargs,
    ) -> np.ndarray:
        alpha = int(kwargs.get("alpha", 16))
        view_type = kwargs.get("view_type", "segments")

        # Load and resize to the requested resolution
        img = Image.open(image_path).convert("RGB")
        img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
        frame = np.array(img)  # (H, W, 3), uint8

        # Run the SVDCodes pipeline
        pixels = get_pixels(frame)
        h = get_transformed_data(pixels)
        codes = get_codes(h, alpha)

        if view_type == "mosaic view":
            segmented = get_segments_mosaic_view(frame, pixels, codes)
        else:
            segmented = get_segments(frame, pixels, codes)

        # Derive a boolean mask: pixels that are NOT completely black are
        # considered foreground.  This gives a sensible binary mask that is
        # compatible with the rest of the segmentation pipeline.
        # Only use the top (resolution × resolution) slice so that a mosaic
        # view (which may be taller) still returns the expected shape.
        mask_frame = segmented[:resolution, :resolution, :]
        mask = np.any(mask_frame > 0, axis=2)  # True where any channel > 0
        return mask

    def segment_image(
        self,
        image_path: str,
        resolution: int,
        **kwargs,
    ) -> np.ndarray:
        """
        Returns the full segmented image array instead of a binary mask.

        This is useful when you want to inspect or visualise the colour
        segments produced by the SVDCodes algorithm directly.

        Returns:
            np.ndarray: Segmented image of shape ``(H, W, 3)`` (or taller
            when ``view_type="mosaic view"``).
        """
        alpha = int(kwargs.get("alpha", 16))
        view_type = kwargs.get("view_type", "segments")

        img = Image.open(image_path).convert("RGB")
        img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
        frame = np.array(img)

        pixels = get_pixels(frame)
        h = get_transformed_data(pixels)
        codes = get_codes(h, alpha)

        if view_type == "mosaic view":
            return get_segments_mosaic_view(frame, pixels, codes)
        return get_segments(frame, pixels, codes)
