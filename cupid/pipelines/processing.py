from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import rembg

from .types import ProcessedImage, CropParameters, CameraPose


class ImageProcessor:
    """Handles image preprocessing including background removal and cropping."""

    def __init__(self, max_size: int = 1024, crop_padding: float = 1.2):
        self.max_size = max_size
        self.crop_padding = crop_padding
        self._rembg_session = None

    @property
    def rembg_session(self):
        if self._rembg_session is None:
            self._rembg_session = rembg.new_session('u2net')
        return self._rembg_session

    def preprocess(self, image: Image.Image) -> ProcessedImage:
        """Remove background if needed and return ProcessedImage."""
        if self._has_alpha(image):
            return ProcessedImage.from_image(image)

        rgb_image = self._resize_if_needed(image.convert('RGB'))
        rgba_image = rembg.remove(rgb_image, session=self.rembg_session)
        return ProcessedImage.from_image(rgba_image)

    def crop_to_content(self, processed: ProcessedImage) -> ProcessedImage:
        """Crop image to bounding box of non-transparent content."""
        image_array = np.array(processed.image)
        max_size = max(processed.image.size)
        alpha = image_array[:, :, 3]

        coords = np.argwhere(alpha > 0.8 * 255)
        bbox = (
            np.min(coords[:, 1]),
            np.min(coords[:, 0]),
            np.max(coords[:, 1]),
            np.max(coords[:, 0]),
        )

        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        size = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * self.crop_padding)

        crop_box = (
            center[0] - size // 2,
            center[1] - size // 2,
            center[0] + size // 2,
            center[1] + size // 2,
        )

        crop_params = CropParameters(
            fov_scale=size / max_size,
            cx_offset=crop_box[0] / max_size,
            cy_offset=crop_box[1] / max_size,
        )

        return ProcessedImage(
            image=processed.image.crop(crop_box),
            crop_params=crop_params,
        )

    def _has_alpha(self, image: Image.Image) -> bool:
        if image.mode != 'RGBA':
            return False
        alpha = np.array(image)[:, :, 3]
        return not np.all(alpha == 255)

    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        max_dim = max(image.size)
        if max_dim <= self.max_size:
            return image
        scale = self.max_size / max_dim
        new_size = (int(image.width * scale), int(image.height * scale))
        return image.resize(new_size, Image.Resampling.LANCZOS)


class ImageEncoder:
    """Encodes images using DINOv2 for conditioning."""

    DINO_SIZE = 518

    def __init__(self, model: nn.Module, visual_cond_resolution: int = 256):
        self.model = model
        self.visual_cond_resolution = visual_cond_resolution
        self.device = None
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    @classmethod
    def from_hub(cls, model_name: str, visual_cond_resolution: int = 256) -> 'ImageEncoder':
        model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        model.eval()
        return cls(model, visual_cond_resolution)

    def to(self, device: torch.device) -> 'ImageEncoder':
        self.device = device
        self.model = self.model.to(device)
        return self

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))

    @torch.no_grad()
    def encode(self, images: List[ProcessedImage]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode images to (dino_features, visual_features)."""
        dino_tensors = []
        visual_tensors = []
        
        for processed in images:
            dino_tensors.append(self._prepare_tensor(processed.image, self.DINO_SIZE))
            visual_tensors.append(self._prepare_tensor(processed.image, self.visual_cond_resolution))

        dino_batch = torch.stack(dino_tensors).to(self.device)
        visual_batch = torch.stack(visual_tensors).to(self.device)

        dino_batch = self.transform(dino_batch)
        features = self.model(dino_batch, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens, visual_batch

    def _prepare_tensor(self, image: Image.Image, size: int) -> torch.Tensor:
        rgba = image.convert('RGBA').resize((size, size), Image.Resampling.LANCZOS)
        alpha = torch.tensor(np.array(rgba.getchannel(3))).float() / 255.0
        rgb = torch.tensor(np.array(rgba.convert('RGB'))).permute(2, 0, 1).float() / 255.0
        return rgb * alpha.unsqueeze(0)


class CameraPoseDecoder:
    """Decodes UV sparse tensors into camera poses."""

    SUPPORTED_METHODS = ('dlt', 'epnp')

    def __init__(self, resolution: int, default_method: str = 'dlt'):
        if default_method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Method must be one of {self.SUPPORTED_METHODS}")
        self.resolution = resolution
        self.default_method = default_method

    def decode(
        self,
        uvs,  # sp.SparseTensor
        method: Optional[str] = None,
    ) -> List[CameraPose]:
        """Decode UV sparse tensor to camera poses."""
        from ..utils.pose_utils import calibrate_camera_dlt, calibrate_camera

        method = method or self.default_method
        poses = []
        for i, uvs_i in enumerate(uvs.unbind(0)):
            xyzs = (uvs_i.coords[:, 1:].contiguous() + 0.5) / self.resolution - 0.5

            if method == 'dlt':
                intrinsic, extrinsic, _ = calibrate_camera_dlt(xyzs.unsqueeze(0), uvs_i.feats.unsqueeze(0))
            elif method == 'epnp':
                intrinsic, extrinsic, _ = calibrate_camera(xyzs.unsqueeze(0), uvs_i.feats.unsqueeze(0))
            else:
                raise ValueError(f"Unknown method: {method}")

            extrinsic_pad = torch.eye(4, dtype=torch.float, device=extrinsic.device)
            extrinsic_pad[:3, :4] = extrinsic[0]
            pose = CameraPose(intrinsic=intrinsic[0], extrinsic=extrinsic_pad)
            poses.append(pose)

        return poses


class SparseStructureDecoder:
    """Decodes latent codes to sparse structure coordinates and UVs."""

    def __init__(
        self,
        structure_decoder: nn.Module,
        uv_decoder: nn.Module,
        predict_ssuv_from_ss: bool = True,
        expand_min_num_activated: int = 10,
        expand_threshold_step: float = 0.05,
    ):
        self.structure_decoder = structure_decoder
        self.uv_decoder = uv_decoder
        self.predict_ssuv_from_ss = predict_ssuv_from_ss
        self.expand_min_num_activated = expand_min_num_activated
        self.expand_threshold_step = expand_threshold_step

    def decode(self, z_s: torch.Tensor):
        """Decode latent to sparse structure with coords and uvs."""
        from ..modules import sparse as sp

        ss_latent_dim = self.structure_decoder.latent_channels
        uv_latent_dim = self.uv_decoder.latent_channels

        ss_logits = self.structure_decoder(z_s[:, :ss_latent_dim])
        suv_logits = self.uv_decoder(z_s[:, ss_latent_dim:ss_latent_dim + uv_latent_dim])

        ss_recon = (ss_logits > 0).int()
        ss_coords = torch.argwhere(ss_recon)[:, [0, 2, 3, 4]].int()

        ssuv_logits, uv_logits = suv_logits[:, :1], suv_logits[:, 1:]

        if self.predict_ssuv_from_ss:
            ssuv_logits, ssuv_recon = ss_logits, ss_recon
        else:
            ssuv_recon = (ssuv_logits > 0).int()

        ssuv_recon_expanded = self._expand_sparse_uv(ssuv_logits, ssuv_recon)

        uv_coords = torch.argwhere(ssuv_recon_expanded)[:, [0, 2, 3, 4]].int()
        uv_values = torch.sigmoid(
            uv_logits[:, :2][
                uv_coords[:, 0], :,
                uv_coords[:, 1], uv_coords[:, 2], uv_coords[:, 3]
            ]
        )

        return {
            'coords': ss_coords,
            'uvs': sp.SparseTensor(feats=uv_values, coords=uv_coords),
        }

    def _expand_sparse_uv(self, logits: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        """Expand UV reconstruction to ensure minimum activated voxels."""
        expanded = recon.clone()
        num_activated = expanded.sum(dim=[1, 2, 3, 4])
        needs_expansion = num_activated < self.expand_min_num_activated

        threshold = 0.0
        while needs_expansion.any():
            threshold -= self.expand_threshold_step
            expanded[needs_expansion] = (logits[needs_expansion] > threshold).int()
            num_activated = expanded.sum(dim=[1, 2, 3, 4])
            needs_expansion = num_activated < self.expand_min_num_activated

        return expanded