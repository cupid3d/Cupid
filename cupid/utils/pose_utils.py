import numpy as np
import cv2
import torch
import torch.nn.functional as F


def calibrate_camera(
    obj_points: torch.Tensor,
    img_points: torch.Tensor,
    initial_camera_matrix: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calibrates the camera for a batch of single-view observations using OpenCV's calibrateCamera.
    Each batch item represents an independent calibration with one view of corresponding 3D object
    points and 2D image points. Assumes image points are in normalized coordinates (0-1 range).

    Args:
        obj_points: torch.Tensor of 3D world points (X, Y, Z) with shape (batch_size, num_points, 3).
        img_points: torch.Tensor of 2D image points (u, v) with shape (batch_size, num_points, 2).
        initial_camera_matrix: Optional torch.Tensor with initial guess for the intrinsic matrix,
            shape (batch_size, 3, 3). If None, a default normalized matrix is used.

    Returns:
        intrinsics: torch.Tensor of recovered intrinsic matrices, shape (batch_size, 3, 3).
        extrinsics: torch.Tensor of extrinsic matrices [R | t], shape (batch_size, 3, 4).
        reproj_errors: torch.Tensor of mean reprojection errors, shape (batch_size,).
    """
    # Convert inputs to NumPy for OpenCV compatibility
    obj_points_np = obj_points.float().contiguous().cpu().numpy() if isinstance(obj_points, torch.Tensor) else obj_points
    img_points_np = img_points.float().contiguous().cpu().numpy() if isinstance(img_points, torch.Tensor) else img_points

    # idx = np.random.choice(obj_points_np.shape[1], 4096, replace=True)
    # obj_points_np = obj_points_np[:, idx]
    # img_points_np = img_points_np[:, idx]

    batch_size = obj_points_np.shape[0]
    
    if initial_camera_matrix is None:
        # Default initial guess for normalized intrinsics (assuming u,v in [0,1])
        default_matrix = np.array([
            [0.8, 0.0, 0.5],
            [0.0, 0.8, 0.5],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        initial_camera_matrix_np = np.tile(default_matrix, (batch_size, 1, 1))
    else:
        initial_camera_matrix_np = initial_camera_matrix.cpu().numpy()

    extrinsics_list = []
    intrinsics_list = []
    reproj_errors_list = []

    for i in range(batch_size):
        # Prepare inputs for single view: lists as required by calibrateCamera
        current_obj_points = [obj_points_np[i]]
        current_img_points = [img_points_np[i]]
        
        # Dummy image size since we're using normalized coordinates and initial guess
        img_size = (1, 1)

        # Perform calibration with fixed distortion parameters (assuming minimal distortion)
        ret, recovered_mtx, recovered_dist, rvecs, tvecs = cv2.calibrateCamera(
            current_obj_points,
            current_img_points,
            img_size,
            cameraMatrix=initial_camera_matrix_np[i],
            distCoeffs=None,
            flags=(
                cv2.CALIB_USE_INTRINSIC_GUESS |
                cv2.CALIB_FIX_K1 |
                cv2.CALIB_FIX_K2 |
                cv2.CALIB_FIX_K3 |
                cv2.CALIB_FIX_K4 |
                cv2.CALIB_FIX_K5 |
                cv2.CALIB_ZERO_TANGENT_DIST
            )
        )

        # Construct extrinsic matrix [R | t]
        rotation_matrix, _ = cv2.Rodrigues(rvecs[0])
        translation_vector = tvecs[0].reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_vector))
        extrinsics_list.append(extrinsic_matrix)

        intrinsics_list.append(recovered_mtx)

        # Compute mean reprojection error for this view
        mean_error = 0.0
        num_views = len(current_obj_points)  # Always 1 in this case
        for view_idx in range(num_views):
            projected_points, _ = cv2.projectPoints(
                current_obj_points[view_idx],
                rvecs[view_idx],
                tvecs[view_idx],
                recovered_mtx,
                recovered_dist
            )

            diff = current_img_points[view_idx] - projected_points.reshape(-1,2)
            per_point_errors = np.linalg.norm(diff, axis=1)
            error = np.mean(per_point_errors)
            mean_error += error
        mean_error /= num_views
        reproj_errors_list.append(mean_error)

    # Stack results and convert back to torch tensors
    intrinsics = torch.from_numpy(np.stack(intrinsics_list)).to(obj_points.device).float()
    extrinsics = torch.from_numpy(np.stack(extrinsics_list)).to(obj_points.device).float()
    reproj_errors = torch.from_numpy(np.stack(reproj_errors_list)).to(obj_points.device).float()

    return intrinsics, extrinsics, reproj_errors


def calibrate_camera_dlt(
    obj_points: torch.Tensor,
    img_points: torch.Tensor,
    initial_camera_matrix: torch.Tensor = None,
    weights: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calibrates the camera for a batch of single-view observations using OpenCV's calibrateCamera.
    Each batch item represents an independent calibration with one view of corresponding 3D object
    points and 2D image points. Assumes image points are in normalized coordinates (0-1 range).

    Args:
        obj_points: torch.Tensor of 3D world points (X, Y, Z) with shape (batch_size, num_points, 3).
        img_points: torch.Tensor of 2D image points (u, v) with shape (batch_size, num_points, 2).
        initial_camera_matrix: Optional torch.Tensor with initial guess for the intrinsic matrix,
            shape (batch_size, 3, 3). If None, a default normalized matrix is used.

    Returns:
        intrinsics: torch.Tensor of recovered intrinsic matrices, shape (batch_size, 3, 3).
        extrinsics: torch.Tensor of extrinsic matrices [R | t], shape (batch_size, 3, 4).
        reproj_errors: torch.Tensor of mean reprojection errors, shape (batch_size,).
    """
    transform = dlt_calibration(obj_points.permute(0,2,1), img_points.permute(0,2,1), weights=weights)
    intrinsics, extrinsics = decompose_projection(transform)
    xyz_cam = obj_points @ transform[:, :3, :3].permute(0,2,1) + transform[:, :3, 3:].permute(0,2,1)
    reproj_errors = ((xyz_cam[..., :2] / xyz_cam[..., 2:]) - img_points).norm(dim=-1).mean(dim=-1)
    return intrinsics, extrinsics, reproj_errors


def decompose_projection(proj_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decomposes a batch of 3x4 projection matrices into intrinsic matrices (K), 
    rotation matrices (R), and translation vectors (t).
    
    Args:
        proj_matrix (torch.Tensor): Batch of projection matrices of shape (B, 3, 4) where P = K [R | t]
    
    Returns:
        K (torch.Tensor): Batch of intrinsic matrices of shape (B, 3, 3).
        R (torch.Tensor): Batch of rotation matrices of shape (B, 3, 3).
        t (torch.Tensor): Batch of translation vectors of shape (B, 3).
    """
    # Ensure input is a batch of 3x4 matrices
    assert proj_matrix.dim() == 3 and proj_matrix.shape[1:] == (3, 4), "Input must be of shape (B, 3, 4)."
    
    # Step 1: Extract the left 3x3 submatrix (M = K * R)
    M = proj_matrix[:, :3, :3].double()
    
    # Step 2: Decompose M into K and R using RQ decomposition
    # Note: PyTorch doesn't have RQ decomposition; use QR and adjust.
    M_inv = torch.linalg.pinv(M)
    Q, R = torch.linalg.qr(M_inv)
    K = torch.linalg.pinv(R).float()
    R = torch.linalg.pinv(Q).float()
    # Step 3: resolve sign ambiguities
    diag = torch.sign(torch.diagonal(K, dim1=-2, dim2=-1))  # Get the sign of the diagonal
    diag[diag == 0] = 1.0  # Set 0 to 1 to avoid division by zero
    D = torch.diag_embed(diag)  # Create a diagonal matrix with the signs
    K = K @ D  # Multiply the intrinsic matrix by the diagonal matrix
    R = D @ R
    
    # Step 4: Compute translation vector t = K^{-1} * P[:, 3]
    t = torch.linalg.pinv(K) @ proj_matrix[:, :, 3:4]
    
    # Step 5
    det = (torch.linalg.det(R) < 0)
    if det.any():
        # should not arrive here because we have normalize proj_matrix
        R[det] *= -1 
    
    # Step 6 normalize scale
    # if (K[:, 2:3, 2:3] < 1e-8).any():
    #     import ipdb; ipdb.set_trace()
    scale = K[:, 2:3, 2:3]

    K = K / scale
    
    # import ipdb; ipdb.set_trace()
    # transfroms = K @ torch.cat([R, t], dim=-1) * scale
    # transform = torch.cat([(K @ R), K @ t], dim=-1)
    # return K, R, t.squeeze(-1), diag, det
    return K, torch.cat([R, t], dim=-1)


def dlt_calibration(obj_points: torch.Tensor, img_points: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute the camera matrix P using the Direct Linear Transformation (DLT) algorithm in PyTorch.
    https://www.ipb.uni-bonn.de/html/teaching/photo12-2021/2021-pho1-21-DLT.pptx.pdf

    Args:
        obj_points: torch.Tensor of 3D world points (X, Y, Z) of shape (batch_size, 3, N) or (1, 3, N)
        img_points: torch.Tensor of 2D image points (u, v) of shape (batch_size, 2, N)
        weights: torch.Tensor of weights (w) of shape (batch_size, N) or None

    Returns:
        P: torch.Tensor, the camera matrix of shape (batch_size, 3, 4)
        A: torch.Tensor, the coefficient matrix of shape (batch_size, 2N, 12)
    """
    # Check input shapes for batched version
    assert obj_points.dim() == 3 and obj_points.shape[1] == 3, "obj_points must be of shape (batch_size, 3, N)"
    assert img_points.dim() == 3 and img_points.shape[1] == 2, "img_points must be of shape (batch_size, 2, N)"
    assert obj_points.shape[2] == img_points.shape[2], "Number of points must match"
    assert weights is None or (weights.dim() == 2 and weights.shape[0] == img_points.shape[0] and weights.shape[1] == img_points.shape[2]), "weights must be of shape (batch_size, N)"
    
    batch_size, _, num_points = img_points.shape
    
    # Ensure the number of input points is at least 6
    if num_points < 6:
        raise ValueError("At least 6 point correspondences are required (N >= 6)")
    
    # Add homogeneous coordinate (1s) to 3D points
    # obj_points: (batch_size, 3, N) -> (batch_size, 4, N)
    obj_points_homo = F.pad(obj_points, (0, 0, 0, 1), mode='constant', value=1)
    
    if weights is not None:
        # weights: (batch_size, N) -> (batch_size, 1, N)
        obj_points_homo = obj_points_homo * weights.unsqueeze(1)
    
    # Separate u and v coordinates
    # img_points: (batch_size, 2, N) -> (batch_size, 1, N) each
    u = img_points[:, 0:1, :]  # (batch_size, 1, N)
    v = img_points[:, 1:2, :]  # (batch_size, 1, N)
    
    # Initialize coefficient matrix A for each batch
    # A: (batch_size, 2N, 12)
    A = torch.zeros(batch_size, 2 * num_points, 12, device=obj_points.device)
    
    # Construct the matrix A for each batch as follows:
    # For each point i:
    #   Row 2i: [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u]
    #   Row 2i+1: [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]
    
    # First 4 columns (block for p11, p12, p13, p14)
    A[:, 0::2, 0:4] = obj_points_homo.transpose(1, 2)  # (batch_size, N, 4)
    
    # Next 4 columns (block for p21, p22, p23, p24)
    A[:, 1::2, 4:8] = obj_points_homo.transpose(1, 2)  # (batch_size, N, 4)
    
    # Last 4 columns (block for p31, p32, p33, p34)
    A[:, 0::2, 8:12] = -u.transpose(1, 2) * obj_points_homo.transpose(1, 2)  # (batch_size, N, 4)
    A[:, 1::2, 8:12] = -v.transpose(1, 2) * obj_points_homo.transpose(1, 2)  # (batch_size, N, 4)
    
    # Solve Ap = 0 using SVD for each batch
    # Use torch.linalg.svd with full_matrices=False for efficiency
    _, _, V = torch.linalg.svd(A, full_matrices=False)
    
    # Solution is the last row of V for each batch (minimizes ||Ap||)
    # V: (batch_size, 2N, 12) -> P_flat: (batch_size, 12)
    P_flat = V[:, -1, :]
    
    # Reshape to camera matrices
    # P: (batch_size, 3, 4)
    P = P_flat.view(batch_size, 3, 4)

    # there is a scale ambiguity (with sign), because we use opencv coordinate
    # the determine should be positive
    P[torch.linalg.det(P[:,:3,:3]) < 0] *= -1

    return P


def camera_parameters_from_frame_data(frame_data: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get camera extrinsic and intrinsic parameters from frame data.
    """
    import utils3d

    c2w = torch.tensor(frame_data['transform_matrix'], dtype=torch.float32)
    c2w[:3, 1:3] *= -1  # Flip Y and Z axes
    extrinsics = torch.inverse(c2w)
    fov = torch.tensor(frame_data['camera_angle_x'])
    intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)

    return extrinsics, intrinsics