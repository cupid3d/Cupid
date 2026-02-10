"""
Mesh Alignment and Composition Module

This module provides functionality for aligning 3D meshes generated from images
using monocular geometry estimation (MoGe) and composing multiple meshes into
a single scene based on their pose transformations.
"""

import os
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from pytorch3d.ops.points_alignment import corresponding_points_alignment
import utils3d

from moge.model.v1 import MoGeModel
from cupid.utils import postprocessing_utils, render_utils


# =============================================================================
# Constants
# =============================================================================

# Transformation matrix to convert from GLB coordinate system (Y-up) to
# Blender/Z-up coordinate system. Represents a +90 degree rotation around X-axis.
# Mapping: X -> X, Y -> Z, Z -> -Y
YUP_TO_ZUP_TRANSFORM = np.array([
    [1,  0,  0,  0],
    [0,  0, -1,  0],
    [0,  1,  0,  0],
    [0,  0,  0,  1]
])


# =============================================================================
# Utility Functions
# =============================================================================

def make_serializable(data):
    """
    Recursively convert tensors and numpy arrays to JSON-serializable types.

    Args:
        data: Input data structure (dict, list, tensor, array, or primitive).

    Returns:
        JSON-serializable version of the input data.
    """
    if isinstance(data, dict):
        return {key: make_serializable(value) for key, value in data.items()}

    if isinstance(data, (list, tuple)):
        return [make_serializable(item) for item in data]

    if isinstance(data, torch.Tensor):
        return data.cpu().numpy().tolist()

    if isinstance(data, np.ndarray):
        return data.tolist()

    if isinstance(data, (np.float32, np.float64)):
        return float(data)

    return data


def _apply_scale_transform(mesh: trimesh.Trimesh, scale) -> None:
    """
    Apply uniform scaling to a mesh in-place.

    Args:
        mesh: The trimesh object to scale.
        scale: Scale factor (float or array-like).
    """
    scale_matrix = np.eye(4)

    if isinstance(scale, (float, int)):
        scale_matrix[:3, :3] *= scale
    else:
        # Handle per-axis scaling if provided as array
        scale_matrix[0, 0] = scale[0] if hasattr(scale, '__len__') else scale
        scale_matrix[1, 1] = scale[1] if hasattr(scale, '__len__') else scale
        scale_matrix[2, 2] = scale[2] if hasattr(scale, '__len__') else scale

    mesh.apply_transform(scale_matrix)


def _erode_mask(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    Apply morphological erosion to a binary mask to remove edge artifacts.

    Args:
        mask: Binary mask tensor of shape (H, W).
        kernel_size: Size of the erosion kernel.

    Returns:
        Eroded binary mask tensor of shape (H, W).
    """
    # Prepare mask for convolution: (H, W) -> (1, 1, H, W)
    mask_4d = mask.unsqueeze(0).unsqueeze(0)

    # Create uniform kernel
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device)

    # Erosion: keep only pixels where all neighbors are foreground
    padding = kernel_size // 2
    full_neighborhood_count = kernel_size * kernel_size
    eroded = F.conv2d(mask_4d, kernel, padding=padding) == full_neighborhood_count

    return eroded.squeeze(0).squeeze(0)


def _create_uv_grid(resolution: int, device: torch.device) -> torch.Tensor:
    """
    Create a normalized UV coordinate grid for unprojection.

    Args:
        resolution: Grid resolution (assumes square grid).
        device: Torch device for tensor allocation.

    Returns:
        UV grid tensor of shape (resolution, resolution, 2).
    """
    # Create normalized coordinates centered at pixel centers
    coords = (torch.arange(0, resolution, dtype=torch.float32, device=device) + 0.5) / resolution
    grid_x, grid_y = torch.meshgrid(coords, coords, indexing='xy')
    return torch.stack([grid_x, grid_y], dim=-1)


# =============================================================================
# Mesh Export Functions
# =============================================================================

def save_mesh(all_outputs, poses, output_dir: str) -> None:
    """
    Export generated meshes to GLB format with associated pose metadata.

    This function converts Gaussian splat and mesh outputs to GLB files,
    applies mesh simplification, and saves pose information for later
    composition.

    Args:
        all_outputs: Single output dict or list of dicts, each containing:
            - 'gaussian': Gaussian splat representation
            - 'mesh': Mesh representation
        poses: Single pose dict or list of dicts, each containing:
            - 'extrinsic': 4x4 extrinsic matrix
            - 'intrinsic': 3x3 intrinsic matrix
            - 'model_scale': Scale factor applied to the model
        output_dir: Directory path for saving outputs.
    """
    # Normalize inputs to list format
    if not isinstance(all_outputs, list):
        all_outputs = [all_outputs]
    if not isinstance(poses, list):
        poses = [poses]

    os.makedirs(output_dir, exist_ok=True)
    glb_paths = []

    # Export each mesh to GLB format
    for idx, outputs in enumerate(all_outputs):
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=0.95,      # Remove 95% of triangles for optimization
            texture_size=1024,  # Texture resolution for the GLB
        )

        glb_path = f'{output_dir}/mesh{idx}.glb'
        glb.export(glb_path)
        glb_paths.append(glb_path)

    # Save metadata JSON with paths and poses
    metadata = {
        'glb_path': glb_paths,
        'pose': make_serializable(poses),
    }

    metadata_path = f'{output_dir}/metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f'Exported aligned meshes and metadata to {output_dir}')


def compose_glbs_with_pose(meta_file: str, output_path: str) -> None:
    """
    Combine multiple GLB meshes into a single scene using pose transformations.

    This function reads mesh files and their associated poses from metadata,
    applies coordinate system transformations and relative pose alignments,
    then exports a unified scene.

    Args:
        meta_file: Path to the metadata JSON file containing GLB paths and poses.
        output_path: Output path for the combined GLB file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(meta_file, 'r') as f:
        meta_data = json.load(f)

    all_meshes = []
    reference_extrinsic = None  # First mesh's extrinsic used as reference frame

    for idx in range(len(meta_data['glb_path'])):
        mesh_path = (Path(meta_file).parent / f'mesh{idx}.glb').as_posix()

        if not os.path.exists(mesh_path):
            print(f"Warning: File not found {mesh_path}, skipping.")
            continue

        # Load GLB file (may be a scene with multiple meshes)
        scene_or_mesh = trimesh.load(mesh_path)
        if isinstance(scene_or_mesh, trimesh.Scene):
            mesh = scene_or_mesh.dump(concatenate=True)
        else:
            mesh = scene_or_mesh

        # Convert from GLB's Y-up to Z-up coordinate system
        # This mimics Blender's import behavior for consistent transformations
        mesh.apply_transform(YUP_TO_ZUP_TRANSFORM)

        # Extract pose information
        pose_data = meta_data['pose'][idx]
        extrinsic = np.array(pose_data['extrinsic'])
        model_scale = pose_data.get('model_scale', 1.0)

        # Apply model scale
        _apply_scale_transform(mesh, model_scale)

        # Apply relative transformation based on first mesh as reference
        if idx == 0:
            reference_extrinsic = extrinsic
            # First mesh stays at origin (identity transform relative to itself)
        else:
            # Transform subsequent meshes relative to the first:
            # T_relative = T_reference^(-1) @ T_current
            relative_transform = np.linalg.inv(reference_extrinsic) @ extrinsic
            mesh.apply_transform(relative_transform)

        all_meshes.append(mesh)
        print(f"Processed mesh {idx}")

    # Compose scene and convert back to GLB coordinate system
    scene = trimesh.Scene(all_meshes)
    zup_to_yup_transform = np.linalg.inv(YUP_TO_ZUP_TRANSFORM)
    scene.apply_transform(zup_to_yup_transform)

    # Export combined scene
    print(f"Exporting combined mesh to {output_path}...")
    scene.export(output_path)
    print("Done.")


# =============================================================================
# Aligner Class
# =============================================================================

class Aligner:
    """
    Aligns generated 3D meshes to monocular depth estimates from input images.

    This class uses the MoGe (Monocular Geometry) model to estimate depth and
    intrinsics from an input image, then computes similarity transformations
    to align rendered mesh depths with the estimated depths.

    Attributes:
        device: Torch device for computation (default: "cuda").
        model: Loaded MoGe model for depth estimation.
        all_outputs: List of 3D generation outputs to align.
        input_image: Input PIL Image for depth estimation.
        resolution: Processing resolution (default: 512).
    """

    def __init__(
        self,
        input_image: Image.Image,
        objects: dict = None,
        resolution: int = 512
    ):
        """
        Initialize the Aligner with an input image and optional outputs.

        Args:
            input_image: PIL Image to use as alignment reference.
            all_outputs: Dict or list of dicts containing generated mesh data.
            resolution: Resolution for depth estimation and alignment.
        """
        self.device = "cuda"
        self.resolution = resolution
        self.input_image = input_image
        self.objects = objects

        # Load MoGe model for monocular depth estimation
        self.model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(self.device)
        self.model.eval()

    def _preprocess_image(self) -> torch.Tensor:
        """
        Preprocess input image for MoGe model inference.

        Returns:
            Preprocessed image tensor of shape (1, 3, H, W).
        """
        # Resize to target resolution (assumes square input)
        assert self.input_image.size[0] == self.input_image.size[1], \
            "Input image must be square"

        resized = self.input_image.resize(
            (self.resolution, self.resolution),
            Image.Resampling.LANCZOS
        ).convert('RGB')

        # Convert to tensor: (H, W, 3) -> (3, H, W) -> (1, 3, H, W)
        tensor = torch.from_numpy(np.array(resized) / 255.0)
        tensor = tensor.to(self.device).permute(2, 0, 1).float()
        return tensor.unsqueeze(0)

    def _compute_pointmaps(
        self,
        src_depth: torch.Tensor,
        src_intrinsic: torch.Tensor,
        tgt_depth: torch.Tensor,
        tgt_intrinsic: torch.Tensor,
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 3D point clouds from depth maps for alignment.

        Args:
            src_depth: Source (rendered mesh) depth map.
            src_intrinsic: Source camera intrinsic matrix.
            tgt_depth: Target (MoGe estimated) depth map.
            tgt_intrinsic: Target camera intrinsic matrix.
            mask: Valid pixel mask for correspondence.

        Returns:
            Tuple of (source_points, target_points) tensors.
        """
        identity_extrinsic = torch.eye(4, device=self.device)
        uv_grid = _create_uv_grid(self.resolution, self.device)

        # Unproject depth maps to 3D points and apply mask
        source_points = utils3d.torch.unproject_cv(
            uv_grid, src_depth,
            extrinsics=identity_extrinsic,
            intrinsics=src_intrinsic
        )[mask]

        target_points = utils3d.torch.unproject_cv(
            uv_grid, tgt_depth,
            extrinsics=identity_extrinsic,
            intrinsics=tgt_intrinsic
        )[mask]

        return source_points, target_points

    def align(self) -> list[dict]:
        """
        Compute alignment transformations for all mesh outputs.

        This method estimates depth from the input image using MoGe, renders
        each mesh to obtain depth maps, then computes similarity transformations
        (rotation, translation, scale) that align rendered depths to estimated
        depths.

        Returns:
            List of pose dictionaries, each containing:
                - 'extrinsic': Aligned 4x4 extrinsic matrix
                - 'intrinsic': 3x3 intrinsic matrix from MoGe
                - 'model_scale': Computed scale factor
        """
        # Run MoGe inference on input image
        input_tensor = self._preprocess_image()
        moge_results = self.model.infer(input_tensor)

        # Extract target (MoGe) depth and camera parameters
        tgt_mask = moge_results['mask'][0].float().to(self.device)
        tgt_intrinsic = moge_results['intrinsics'][0].to(self.device)
        tgt_depth = moge_results['depth'][0].to(self.device)

        output_poses = []

        for outputs in self.objects:
            # Render mesh from its original pose to get source depth
            mesh_render = render_utils.render_pose(
                outputs['mesh'][0],
                pose=outputs['pose'][0],
                resolution=self.resolution
            )

            src_intrinsic = outputs['pose'][0]['intrinsic'].to(self.device)
            src_extrinsic = outputs['pose'][0]['extrinsic'].to(self.device)
            src_depth = torch.tensor(
                mesh_render['depth'][0],
                dtype=torch.float32,
                device=self.device
            )
            src_mask = torch.tensor(
                mesh_render['mask'][0] / 255.0,
                dtype=torch.float32,
                device=self.device
            )

            # Erode source mask to remove edge artifacts from rendering
            src_mask = (src_mask > 0).float()
            src_mask = _erode_mask(src_mask, kernel_size=5)

            # Compute intersection mask for valid correspondences
            valid_mask = (src_mask > 0) & (tgt_mask > 0)
            if valid_mask.sum() == 0:
                raise ValueError("No valid correspondences found between meshes!")

            # Compute 3D point clouds for alignment
            source_points, target_points = self._compute_pointmaps(
                src_depth, src_intrinsic,
                tgt_depth, tgt_intrinsic,
                valid_mask
            )

            # Compute similarity transformation (Procrustes alignment)
            similarity_result = corresponding_points_alignment(
                source_points.unsqueeze(0),
                target_points.unsqueeze(0),
                estimate_scale=True
            )

            rotation = similarity_result.R[0]      # (3, 3)
            translation = similarity_result.T[0]  # (3,)
            scale = similarity_result.s[0]        # scalar

            # Construct transformation matrix from model to view space
            model_to_view = torch.eye(4, device=self.device)
            model_to_view[:3, :3] = rotation.t()
            model_to_view[:3, 3] = translation

            # Apply scale to extrinsic translation and compose final transform
            scaled_extrinsic = src_extrinsic.clone()
            scaled_extrinsic[:3, 3] = src_extrinsic[:3, 3] * scale
            aligned_extrinsic = model_to_view @ scaled_extrinsic

            output_poses.append({
                'extrinsic': aligned_extrinsic.cpu(),
                'intrinsic': tgt_intrinsic.cpu(),
                'model_scale': scale,
            })

        return output_poses

    def export_meshes(self, output_dir: str) -> None:
        """
        Align meshes and export them with pose metadata.

        Args:
            output_dir: Directory path for saving aligned meshes and metadata.
        """
        aligned_poses = self.align()
        save_mesh(self.objects, aligned_poses, output_dir)

    @staticmethod
    def compose_mesh_from_metadata(meta_file: str, output_path: str) -> None:
        """
        Combine meshes referenced in metadata into a single GLB file.

        Args:
            meta_file: Path to metadata JSON file.
            output_path: Output path for the combined GLB.
        """
        compose_glbs_with_pose(meta_file, output_path)