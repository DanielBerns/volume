import numpy as np
from PIL import Image

def create_binary_mask(image_path: str, resolution: int, is_contrasting_bg: bool = True) -> np.ndarray:
    """
    Loads an image using Pillow, resizes it, and generates a boolean numpy mask.
    True (1) represents the plant, False (0) represents empty space.
    """
    # Load image and resize to the target 3D grid resolution
    img = Image.open(image_path).convert('HSV')
    img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)

    # Convert to a numpy array for vectorized thresholding
    hsv_array = np.array(img)

    # Note: These exact thresholds will need to be calibrated based on
    # your lighting, the specific plant's color, and the background material.
    if is_contrasting_bg:
        # Example: Assuming we want to isolate green plant material.
        # We extract based on a Hue range. Green hue in Pillow HSV (0-255) is roughly 40-100.
        hue_channel = hsv_array[:, :, 0]
        plant_mask = (hue_channel > 40) & (hue_channel < 100)
    else:
        # For the third image lacking the contrasting background,
        # you might rely on saturation, brightness, or edge detection.
        # Placeholder: Simple saturation threshold
        saturation_channel = hsv_array[:, :, 1]
        plant_mask = saturation_channel > 50

    return plant_mask

def estimate_volume(img_xy_path: str, img_yz_path: str, img_xz_path: str, resolution: int = 256) -> float:
    """
    Calculates the intersection of three orthogonal 2D masks to estimate the 3D volume
    inside a 1 cubic meter space.
    """
    # 1. Generate 2D masks (N x N)
    # We assume img_xy and img_yz have the contrasting background, and img_xz does not.
    mask_xy = create_binary_mask(img_xy_path, resolution, is_contrasting_bg=True)
    mask_yz = create_binary_mask(img_yz_path, resolution, is_contrasting_bg=True)
    mask_xz = create_binary_mask(img_xz_path, resolution, is_contrasting_bg=False)

    # 2. Voxel Intersection via NumPy Broadcasting
    # Reshape the 2D (N x N) masks to project them into a 3D (N x N x N) space:
    # vol_xy extrudes along the z-axis
    vol_xy = mask_xy[:, :, np.newaxis]  # Shape: (N, N, 1)

    # vol_yz extrudes along the x-axis
    vol_yz = mask_yz[np.newaxis, :, :]  # Shape: (1, N, N)

    # vol_xz extrudes along the y-axis
    vol_xz = mask_xz[:, np.newaxis, :]  # Shape: (N, 1, N)

    # The visual hull is the logical AND of all three extruded masks
    voxel_grid = vol_xy & vol_yz & vol_xz

    # 3. Volume Calculation
    # Count the total number of 'True' voxels (plant material)
    filled_voxels = np.sum(voxel_grid)

    # Calculate the physical volume
    # The total space is 1 cubic meter, divided into resolution^3 voxels
    total_voxels = resolution ** 3
    voxel_volume_m3 = 1.0 / total_voxels

    estimated_volume_m3 = filled_voxels * voxel_volume_m3

    return estimated_volume_m3

# Example of how it will be called by your backend:
# volume = estimate_volume("data/run_1/photo_xy.jpg", "data/run_1/photo_yz.jpg", "data/run_1/photo_xz.jpg")
# print(f"Estimated Volume: {volume:.6f} m³")
