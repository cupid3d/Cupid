from typing import List, Dict, Any, Optional, Union
import torch
import torch.nn as nn
from PIL import Image
import utils3d

from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from .types import ProcessedImage, CameraPose
from .processing import (
    ImageProcessor,
    ImageEncoder,
    CameraPoseDecoder,
    SparseStructureDecoder,
)


class Cupid3DPipeline(Pipeline):
    """
    Pipeline for inferring Cupid image-to-3D models.

    Args:
        models: Dictionary of model components.
        sparse_structure_sampler: Sampler for sparse structures.
        slat_sampler: Sampler for structured latents.
        slat_normalization: Normalization parameters for slat.
        image_cond_model: Name of the image conditioning model.
        visual_cond_resolution: Resolution for visual conditioning.
        predict_ssuv_from_ss: Whether to predict UV from sparse structure.
        ssuv_expand_min_num_activated: Minimum activated voxels for UV.
        ssuv_expand_threshold_step: Threshold step for UV expansion.
        default_pose_solver: Default camera pose solver method.
    """

    def __init__(
        self,
        models: Dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: Dict = None,
        image_cond_model: str = None,
        visual_cond_resolution: int = 256,
        predict_ssuv_from_ss: bool = True,
        ssuv_expand_min_num_activated: int = 10,
        ssuv_expand_threshold_step: float = 0.05,
        default_pose_solver: str = "dlt",
    ):
        self.visual_cond_resolution = visual_cond_resolution
        self.predict_ssuv_from_ss = predict_ssuv_from_ss
        self.ssuv_expand_min_num_activated = ssuv_expand_min_num_activated
        self.ssuv_expand_threshold_step = ssuv_expand_threshold_step
        self.default_pose_solver = default_pose_solver

        # Initialize processors
        self.image_processor = ImageProcessor()
        self.image_encoder: Optional[ImageEncoder] = None
        self.structure_decoder: Optional[SparseStructureDecoder] = None
        self.pose_decoder: Optional[CameraPoseDecoder] = None

        # Sampler state
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization

        if models is None:
            return

        super().__init__(models)
        self._init_image_cond_model(image_cond_model)
        self._init_decoders()

    def _init_image_cond_model(self, name: str):
        """Initialize the image conditioning model."""
        self.image_encoder = ImageEncoder.from_hub(name, self.visual_cond_resolution)
        self.image_encoder.to(self.device)
        self.models['image_cond_model'] = self.image_encoder

    def _init_decoders(self):
        """Initialize decoder components."""
        self.structure_decoder = SparseStructureDecoder(
            structure_decoder=self.models['sparse_structure_decoder'],
            uv_decoder=self.models['sparse_structure_uv_decoder'],
            predict_ssuv_from_ss=self.predict_ssuv_from_ss,
            expand_min_num_activated=self.ssuv_expand_min_num_activated,
            expand_threshold_step=self.ssuv_expand_threshold_step,
        )

        resolution = self.models['slat_flow_model'].resolution
        self.pose_decoder= CameraPoseDecoder(resolution, self.default_pose_solver)

    @staticmethod
    def from_pretrained(path: str, cls=None) -> 'Cupid3DPipeline':
        """Load a pretrained model from local path or Hugging Face repository."""
        pipeline = super(
            Cupid3DPipeline,
            Cupid3DPipeline,
        ).from_pretrained(path)

        new_pipeline = (cls or Cupid3DPipeline)()
        new_pipeline.__dict__.update(pipeline.__dict__)
        args = pipeline._pretrained_args

        # Setup samplers
        ss_config = args['sparse_structure_sampler']
        new_pipeline.sparse_structure_sampler = getattr(samplers, ss_config['name'])(**ss_config['args'])
        new_pipeline.sparse_structure_sampler_params = ss_config['params']

        slat_config = args['slat_sampler']
        new_pipeline.slat_sampler = getattr(samplers, slat_config['name'])(**slat_config['args'])
        new_pipeline.slat_sampler_params = slat_config['params']

        new_pipeline.slat_normalization = args['slat_normalization']
        new_pipeline._init_image_cond_model(args['image_cond_model'])
        new_pipeline._init_decoders()

        return new_pipeline

    # ==================== Image Processing ====================

    def preprocess_image(self, image: Image.Image) -> ProcessedImage:
        """Preprocess input image (remove background if needed)."""
        return self.image_processor.preprocess(image)

    def crop_image(self, processed: ProcessedImage) -> ProcessedImage:
        """Crop image to content bounding box."""
        return self.image_processor.crop_to_content(processed)

    
    def get_cond(self, images: List[ProcessedImage]) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get conditioning information for the model."""
        cond, image_raw = self.image_encoder.encode(images)
        # self.encode_image(images)
        return {
            'cond': cond,
            'neg_cond': torch.zeros_like(cond),
        }, image_raw

    # ==================== Decoding ====================

    def decode_zs(self, z_s: torch.Tensor) -> Dict[str, Any]:
        """Decode latent z_s to sparse structure coords and UVs."""
        return self.structure_decoder.decode(z_s)

    def decode_uv(
        self,
        uvs: sp.SparseTensor,
        pose_solver: Optional[str] = None,
    ) -> List[CameraPose]:
        """Decode UV sparse tensor to camera poses."""
        return self.pose_decoder.decode(uvs, pose_solver)

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = None,
    ) -> Dict[str, List]:
        """Decode structured latent to output formats."""
        formats = formats or ['mesh', 'gaussian', 'radiance_field']
        result = {}

        decoder_map = {
            'mesh': 'slat_decoder_mesh',
            'gaussian': 'slat_decoder_gs',
            'radiance_field': 'slat_decoder_rf',
        }

        for fmt in formats:
            decoder_name = decoder_map.get(fmt)
            if decoder_name and decoder_name in self.models:
                result[fmt] = [self.models[decoder_name](latent)[0] for latent in slat]

        return result

    # ==================== Sampling ====================

    def sample_sparse_structure(
        self,
        cond: Dict[str, torch.Tensor],
        num_samples: int = 1,
        sampler_params: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Sample sparse structures with the given conditioning."""
        flow_model = self.models['sparse_structure_flow_model']
        res = flow_model.resolution
        in_channels = flow_model.in_channels

        noise = torch.randn(num_samples, in_channels, res, res, res).to(self.device)
        params = {**self.sparse_structure_sampler_params, **(sampler_params or {})}

        result = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **params,
            verbose=True,
        )

        return self.decode_zs(result.samples)

    def sample_slat(
        self,
        cond: Dict[str, torch.Tensor],
        coords: torch.Tensor,
        sampler_params: Dict[str, Any] = None,
    ) -> sp.SparseTensor:
        """Sample structured latent with the given conditioning."""
        flow_model = self.models['slat_flow_model']

        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.out_channels).to(self.device),
            coords=coords,
        )

        # Compute UVs from pose
        uvs = self._make_sparse_uvs_from_pose(noise, cond['extrinsic'], cond['intrinsic'])

        model_cond = {
            'visual_cond': cond['visual_cond'],
            'dino_cond': cond['cond'],
        }
        neg_cond = {
            'visual_cond': torch.zeros_like(cond['visual_cond']),
            'dino_cond': torch.zeros_like(cond['cond']),
        }

        params = {**self.slat_sampler_params, **(sampler_params or {})}

        result = self.slat_sampler.sample(
            flow_model,
            noise,
            uvs=uvs,
            cond=model_cond,
            neg_cond=neg_cond,
            **params,
            verbose=True,
        )

        # Denormalize
        std = torch.tensor(self.slat_normalization['std'])[None].to(result.samples.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(result.samples.device)
        return result.samples * std + mean

    def _make_sparse_uvs_from_pose(
        self,
        x: sp.SparseTensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> sp.SparseTensor:
        """Project 3D coordinates to UV space using camera poses."""
        res = self.models['slat_flow_model'].resolution
        uvs_list = []

        for i, x_i in enumerate(x.unbind(0)):
            xyzs = (x_i.coords[:, 1:].float() + 0.5) / res - 0.5
            uvs, _ = utils3d.torch.project_cv(xyzs, intrinsics=intrinsics[i], extrinsics=extrinsics[i])
            uvs_list.append(uvs)

        uvs = torch.cat(uvs_list, dim=0).clamp(0.0, 1.0)
        return sp.SparseTensor(feats=uvs, coords=x.coords)

    # ==================== Main Entry Point ====================

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: Dict[str, Any] = None,
        slat_sampler_params: Dict[str, Any] = None,
        formats: List[str] = None,
        preprocess_image: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full pipeline.

        Args:
            image: Input image.
            num_samples: Number of samples to generate.
            seed: Random seed.
            sparse_structure_sampler_params: Sampler params for sparse structure.
            slat_sampler_params: Sampler params for structured latent.
            formats: Output formats ('mesh', 'gaussian', 'radiance_field').
            preprocess_image: Whether to crop the image.

        Returns:
            Dict containing coords, uvs, and decoded outputs.
        """
        torch.manual_seed(seed)
        formats = formats or ['mesh', 'gaussian', 'radiance_field']

        # add mask if provided
        if mask is not None:
            image = image.copy()
            image.putalpha(mask.split()[-1])

        processed = self.preprocess_image(image)
        if preprocess_image:
            processed = self.crop_image(processed)

        # Stage 1: Predict coarse structure and camera pose
        cond, visual_feat = self.get_cond([processed])
        structure_output = self.sample_sparse_structure(
            cond, num_samples, sparse_structure_sampler_params
        )
        poses = self.decode_uv(structure_output['uvs'])

        # Stage 2: Sample structured latent
        slat_cond = {
            **cond,
            'extrinsic': torch.stack([p.extrinsic for p in poses]).to(self.device),
            'intrinsic': torch.stack([p.intrinsic for p in poses]).to(self.device),
            'visual_cond': visual_feat.repeat(num_samples, 1, 1, 1),
        }
        slat = self.sample_slat(slat_cond, structure_output['coords'], slat_sampler_params)

        # Adjust poses for original image dimensions
        adjusted_poses = [pose.de_crop(processed.crop_params).as_dict() for pose in poses]

        return {
            'pose': adjusted_poses,
            **self.decode_slat(slat, formats),
        }