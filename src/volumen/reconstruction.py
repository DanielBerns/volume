import numpy as np

def compute_intrinsic_matrix(focal_length_35mm: float, resolution: int) -> np.ndarray:
    """
    Computes a simple pinhole intrinsic matrix based on 35mm equivalent focal length.
    Assuming square pixels and the optical center is at the image center.
    """
    # 35mm film is 36mm wide.
    sensor_width = 36.0
    
    # f_px = focal_length_mm * (image_width_px / sensor_width_mm)
    f_px = focal_length_35mm * (resolution / sensor_width)
    
    cx = resolution / 2.0
    cy = resolution / 2.0
    
    K = np.array([
        [f_px, 0.0, cx],
        [0.0, f_px, cy],
        [0.0, 0.0, 1.0]
    ])
    return K

def compute_extrinsic_matrix(cam_pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Computes a 4x4 view matrix using lookAt method.
    """
    z_axis = target - cam_pos
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    x_axis = np.cross(up, z_axis)
    # Handle collinear up and z
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross(np.array([1.0, 0.0, 0.0]), z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    R = np.vstack([x_axis, y_axis, z_axis])
    t = -R @ cam_pos
    
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = t
    return E

def perspective_visual_hull(
    mask_xy: np.ndarray, 
    mask_yz: np.ndarray, 
    mask_xz: np.ndarray,
    focal_length: float = 50.0,
    resolution: int = 256,
    camera_distance: float = 1.5,
    box_size: float = 1.0
) -> float:
    """
    Computes volume using perspective projection.
    """
    # 1. Create voxel grid (centered at origin)
    x = np.linspace(-box_size/2, box_size/2, resolution)
    y = np.linspace(-box_size/2, box_size/2, resolution)
    z = np.linspace(-box_size/2, box_size/2, resolution)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Flatten grid points to shape (4, N^3)
    pts_3d = np.vstack((X.ravel(), Y.ravel(), Z.ravel(), np.ones_like(X.ravel())))
    
    # 2. Setup Cameras
    K = compute_intrinsic_matrix(focal_length, resolution)
    # Projection matrix P = K [R|t]
    # K extends to 3x4 to multiply with 4x4 E
    K_proj = np.hstack((K, np.zeros((3, 1))))
    
    # Cam XY (looking along Z axis)
    E_xy = compute_extrinsic_matrix(
        cam_pos=np.array([0.0, 0.0, -camera_distance]),
        target=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, -1.0, 0.0]) # image Y is down
    )
    P_xy = K_proj @ E_xy
    
    # Cam YZ (looking along X axis)
    E_yz = compute_extrinsic_matrix(
        cam_pos=np.array([-camera_distance, 0.0, 0.0]),
        target=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, -1.0, 0.0])
    )
    P_yz = K_proj @ E_yz
    
    # Cam XZ (looking along Y axis)
    E_xz = compute_extrinsic_matrix(
        cam_pos=np.array([0.0, -camera_distance, 0.0]),
        target=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 0.0, 1.0]) # Note: up vector changed to avoid collinearity depending on orientation
    )
    P_xz = K_proj @ E_xz
    
    def project_and_sample(P, mask):
        pts_2d = P @ pts_3d
        # Perspective divide
        pts_2d = pts_2d[:2, :] / pts_2d[2, :]
        
        u = np.round(pts_2d[0, :]).astype(int)
        v = np.round(pts_2d[1, :]).astype(int)
        
        valid = (u >= 0) & (u < resolution) & (v >= 0) & (v < resolution)
        
        # Initialize result array
        res = np.zeros_like(valid)
        
        # Use advanced indexing only for valid coordinates
        # Note: mask is assumed to be (resolution, resolution) -> (height, width) -> (v, u)
        res[valid] = mask[v[valid], u[valid]]
        return res
        
    in_xy = project_and_sample(P_xy, mask_xy)
    in_yz = project_and_sample(P_yz, mask_yz)
    in_xz = project_and_sample(P_xz, mask_xz)
    
    voxel_grid = in_xy & in_yz & in_xz
    
    filled_voxels = np.sum(voxel_grid)
    voxel_volume_m3 = (box_size / resolution) ** 3
    estimated_volume_m3 = filled_voxels * voxel_volume_m3
    
    return estimated_volume_m3
