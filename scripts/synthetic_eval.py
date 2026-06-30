import sys
import os
import numpy as np

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from volumen.reconstruction import perspective_visual_hull, compute_intrinsic_matrix, compute_extrinsic_matrix

def generate_synthetic_plant(resolution: int = 128, radius: float = 0.05, box_size: float = 1.0) -> np.ndarray:
    """
    Generates a 3D boolean voxel grid containing a simple synthetic plant (a trunk with branches).
    """
    grid = np.zeros((resolution, resolution, resolution), dtype=bool)
    
    # Coordinate grid
    x = np.linspace(-box_size/2, box_size/2, resolution)
    y = np.linspace(-box_size/2, box_size/2, resolution)
    z = np.linspace(-box_size/2, box_size/2, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    def add_cylinder(start, end, r):
        start = np.array(start)
        end = np.array(end)
        vec = end - start
        length = np.linalg.norm(vec)
        if length == 0: return
        dir = vec / length
        
        # Vectorized distance from segment
        # P = (X, Y, Z)
        # t = dot(P - start, dir)
        # t = clamped to [0, length]
        # dist = norm(P - (start + t*dir))
        
        P_minus_start_x = X - start[0]
        P_minus_start_y = Y - start[1]
        P_minus_start_z = Z - start[2]
        
        t = P_minus_start_x * dir[0] + P_minus_start_y * dir[1] + P_minus_start_z * dir[2]
        t = np.clip(t, 0, length)
        
        proj_x = start[0] + t * dir[0]
        proj_y = start[1] + t * dir[1]
        proj_z = start[2] + t * dir[2]
        
        dist_sq = (X - proj_x)**2 + (Y - proj_y)**2 + (Z - proj_z)**2
        grid[dist_sq <= r**2] = True

    # Main trunk
    add_cylinder([0, -0.4, 0], [0, 0.4, 0], radius)
    # Branches
    add_cylinder([0, 0.0, 0], [0.3, 0.3, 0.2], radius * 0.7)
    add_cylinder([0, -0.1, 0], [-0.3, 0.2, -0.2], radius * 0.7)
    add_cylinder([0, 0.2, 0], [0.0, 0.4, -0.3], radius * 0.6)

    return grid

def render_mask(grid_3d: np.ndarray, P: np.ndarray, resolution: int, box_size: float) -> np.ndarray:
    """
    Renders a 2D boolean mask from a 3D boolean grid using projection matrix P.
    """
    # Get coordinates of True voxels
    x = np.linspace(-box_size/2, box_size/2, resolution)
    y = np.linspace(-box_size/2, box_size/2, resolution)
    z = np.linspace(-box_size/2, box_size/2, resolution)
    
    # Non-zero indices
    idx_x, idx_y, idx_z = np.nonzero(grid_3d)
    
    # 3D points
    pts_3d = np.vstack((x[idx_x], y[idx_y], z[idx_z], np.ones_like(idx_x)))
    
    # Project
    pts_2d = P @ pts_3d
    pts_2d = pts_2d[:2, :] / pts_2d[2, :]
    
    u = np.round(pts_2d[0, :]).astype(int)
    v = np.round(pts_2d[1, :]).astype(int)
    
    valid = (u >= 0) & (u < resolution) & (v >= 0) & (v < resolution)
    u = u[valid]
    v = v[valid]
    
    mask = np.zeros((resolution, resolution), dtype=bool)
    mask[v, u] = True
    
    # A bit of morphological dilation to fill holes caused by point projection
    from scipy.ndimage import binary_dilation
    mask = binary_dilation(mask, iterations=2)
    
    return mask

def main():
    print("Generating synthetic 3D plant...")
    res = 128
    box_size = 1.0
    grid = generate_synthetic_plant(resolution=res, radius=0.05, box_size=box_size)
    
    gt_filled = np.sum(grid)
    voxel_vol = (box_size / res) ** 3
    gt_volume = gt_filled * voxel_vol
    
    print(f"Ground Truth Volume: {gt_volume:.6f} m³")
    
    # Cameras
    focal_length = 50.0
    camera_distance = 1.5
    K = compute_intrinsic_matrix(focal_length, res)
    K_proj = np.hstack((K, np.zeros((3, 1))))
    
    E_xy = compute_extrinsic_matrix(
        cam_pos=np.array([0.0, 0.0, -camera_distance]),
        target=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, -1.0, 0.0])
    )
    E_yz = compute_extrinsic_matrix(
        cam_pos=np.array([-camera_distance, 0.0, 0.0]),
        target=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, -1.0, 0.0])
    )
    E_xz = compute_extrinsic_matrix(
        cam_pos=np.array([0.0, -camera_distance, 0.0]),
        target=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 0.0, 1.0])
    )
    
    P_xy = K_proj @ E_xy
    P_yz = K_proj @ E_yz
    P_xz = K_proj @ E_xz
    
    print("Rendering 2D masks...")
    mask_xy = render_mask(grid, P_xy, res, box_size)
    mask_yz = render_mask(grid, P_yz, res, box_size)
    mask_xz = render_mask(grid, P_xz, res, box_size)
    
    print("Running volume estimation algorithm...")
    estimated_vol = perspective_visual_hull(
        mask_xy, mask_yz, mask_xz,
        focal_length=focal_length,
        resolution=res,
        camera_distance=camera_distance,
        box_size=box_size
    )
    
    print(f"Estimated Volume:    {estimated_vol:.6f} m³")
    
    # Overestimation is expected for Visual Hulls due to concavities.
    error = abs(estimated_vol - gt_volume) / gt_volume * 100
    print(f"Error:               {error:.2f}%")

if __name__ == "__main__":
    main()
