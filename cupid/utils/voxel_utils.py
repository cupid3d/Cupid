import numpy as np
import torch
from PIL import Image


def decode_depth_map(depth_img, frame_data, background_depth=-np.inf):
    """
    Decode depth map from depth image and renormalize it to 3D space.
    
    Args:
        depth_img: (H, W) array of depth values in pixels
        frame_data: frame data containing depth min/max values
        background_depth: depth value to set for background pixels
        
    Returns:
        depth_m: (H, W) array of depth values in meters
    """
    depth_raw = np.array(depth_img, dtype=np.uint16)
    
    # Build invalid mask for background pixels
    invalid_mask = (depth_raw == 65535)
    
    # Normalize to [0,1]
    depth_norm = depth_raw.astype(np.float32) / 65535.0
    
    # Get depth range from frame data
    d_min, d_max = frame_data['depth']['min'], frame_data['depth']['max']
    
    # Linear remap to [d_min, d_max]
    depth_m = depth_norm * (d_max - d_min) + d_min
    
    # Invalidate background
    depth_m[invalid_mask] = background_depth
 
    return depth_m


def read_and_decode_depth_map(depth_map_path, frame_data, background_depth=-np.inf):
    """
    Read depth map from file and renormalize it to 3D space.
    
    Args:
        depth_map_path: path to the depth PNG file
        frame_data: frame data containing depth min/max values
        background_depth: depth value to set for background pixels
        
    Returns:
        depth_m: (H, W) array of depth values in meters
    """
    # Load and decode depth map
    depth_img = Image.open(depth_map_path)
    return decode_depth_map(depth_img, frame_data, background_depth)


def dilate_sparse_grid(coords: torch.Tensor, resolution: int, sigma: float, min_weight: float):
    """
    Perform dilation on sparse 3D grid using Gaussian kernel.
    
    Args:
        coords: torch.Tensor (N, 3) - original voxel coordinates in [0, resolution-1]
        resolution: int - grid resolution
        sigma: float - standard deviation of Gaussian kernel
        min_weight: float - minimum weight to include in the dilated grid
        
    Returns:
        dilated_coords: torch.Tensor (M, 3) - expanded coordinates
        weights: torch.Tensor (M,) - corresponding weights
    """
    assert coords.ndim == 2 and coords.size(1) == 3

    # Create Gaussian kernel for the radius calculated from sigma
    chi2_value = 7.814728  # 95% percentile of chi2 distribution with 3 degrees of freedom
    r = int(np.ceil(sigma * np.sqrt(chi2_value)))
    coord_range = torch.arange(-r, r + 1, device=coords.device, dtype=coords.dtype)

    # Create 3D grid for kernel
    X, Y, Z = torch.meshgrid(coord_range, coord_range, coord_range, indexing='ij')
    offsets = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1) # (kernel_size^3, 3)
    dist_squared = offsets.float().square().sum(dim=1) # (kernel_size^3,)
    kernel = torch.exp(-dist_squared / (2 * sigma**2))

    # Filter out low-weight offsets
    kernel_mask = kernel >= min_weight
    kernel = kernel[kernel_mask]
    offsets = offsets[kernel_mask]
    
    # Apply all offsets to all coordinates at once and filter out invalid coordinates
    all_new_coords = coords[:, None, :] + offsets[None, :, :]  # (N, num_offsets, 3)
    valid = ((all_new_coords >= 0) & (all_new_coords < resolution)).all(dim=2)  # (N, num_offsets)
    
    # Apply mask to get flattened valid coordinates and weights
    valid_coords = all_new_coords[valid]
    valid_weights = kernel[None, :].expand(coords.size(0), -1)[valid]

    # Linearize coords as key to remove duplicates and aggregate weights
    keys = valid_coords[:, 0] * resolution**2 + valid_coords[:, 1] * resolution + valid_coords[:, 2]
    unique_keys, inverse_indices = torch.unique(keys, sorted=True, return_inverse=True)
    
    # Aggregate weights for duplicate coordinates
    weights = torch.full((unique_keys.numel(),), 0.0, device=coords.device, dtype=torch.float32)
    weights.scatter_reduce_(0, inverse_indices, valid_weights, reduce="amax", include_self=False)

    # Recover (x,y,z)
    x = (unique_keys // resolution**2)
    y = ((unique_keys // resolution) % resolution)
    z = (unique_keys % resolution)
    dilated_coords = torch.stack([x, y, z], dim=1)
    
    return dilated_coords, weights