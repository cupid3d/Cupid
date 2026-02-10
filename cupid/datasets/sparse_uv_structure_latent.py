import os
import json
from typing import *
import numpy as np
import torch
import utils3d
from ..representations.octree import DfsOctree as Octree
from ..renderers import OctreeRenderer
from .components import ImageConditionedMixin, TransformConditionedMixin
from .sparse_structure_latent import SparseStructureLatent
from .sparse_uv_structure import SparseUVStructure
from .. import models


class SparseUVStructureLatentVisMixin:
    def __init__(
        self,
        *args,
        pretrained_suv_dec: Optional[str] = None,
        suv_dec_path: Optional[str] = None,
        suv_dec_ckpt: Optional[str] = None,
        suv_latent_channels: Optional[int] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.suv_dec = None
        self.pretrained_suv_dec = pretrained_suv_dec
        self.suv_dec_path = suv_dec_path
        self.suv_dec_ckpt = suv_dec_ckpt
        self.suv_latent_channels = suv_latent_channels
        if pretrained_suv_dec is None and (suv_dec_path is None or suv_dec_ckpt is None):
            raise ValueError('pretrained_suv_dec or suv_dec_path and suv_dec_ckpt must be provided')
        if self.suv_latent_channels is None and self.suv_dec_path is not None:
            cfg = json.load(open(os.path.join(self.suv_dec_path, 'config.json'), 'r'))
            self.suv_latent_channels = cfg['models']['decoder']['args']['latent_channels']
        
    def _loading_suv_dec(self):
        if self.suv_dec is not None:
            return
        if self.suv_dec_path is not None:
            cfg = json.load(open(os.path.join(self.suv_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.suv_dec_path, 'ckpts', f'decoder_{self.suv_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_suv_dec)
        self.suv_dec = decoder.cuda().eval()

    def _delete_suv_dec(self):
        if self.suv_dec is not None:
            del self.suv_dec
            self.suv_dec = None

    @torch.no_grad()
    def decode_suv_latent(self, suv_z, batch_size=4):
        """Decode UV structure latent to get ssuv and uv_volume."""
        self._loading_suv_dec()
        
        if self.normalization is not None:
            suv_z = suv_z * self.std.to(suv_z.device) + self.mean.to(suv_z.device)
        
        logits = []
        for i in range(0, suv_z.shape[0], batch_size):
            logits.append(self.suv_dec(suv_z[i:i+batch_size]))
        logits = torch.cat(logits, dim=0)
        
        # Split the decoder output: ssuv (1 channel) + uv_volume (2 channels)
        ssuv_logits = logits[:, :1, :, :, :]  # [N, 1, H, W, D]
        uv_logits = logits[:, 1:, :, :, :]    # [N, 2, H, W, D]
        
        # Convert logits to binary predictions and UV volume
        ssuv = (ssuv_logits > 0).float()
        uv_volume = torch.sigmoid(uv_logits)
        
        self._delete_suv_dec()
        return ssuv, uv_volume

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[torch.Tensor, dict]):
        x_0 = x_0 if isinstance(x_0, torch.Tensor) else x_0['x_0']
        ssuv, uv_volume = self.decode_suv_latent(x_0.cuda())
        
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = 'voxel'
        
        # Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        exts = []
        ints = []
        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        images = []
        
        # Build each representation
        for i in range(ssuv.shape[0]):
            # Render UV structure
            representation_suv = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device='cuda',
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )
            coords_suv = torch.nonzero(ssuv[i, 0] > 0, as_tuple=False)
            resolution = ssuv.shape[-1]
            representation_suv.position = coords_suv.float() / resolution
            representation_suv.depth = torch.full((representation_suv.position.shape[0], 1), int(np.log2(resolution)), dtype=torch.uint8, device='cuda')

            # Extract UV values from uv_volume at these coordinates
            if coords_suv.shape[0] > 0:
                u_values = uv_volume[i, 0, coords_suv[:, 0], coords_suv[:, 1], coords_suv[:, 2]]
                v_values = uv_volume[i, 1, coords_suv[:, 0], coords_suv[:, 1], coords_suv[:, 2]]
                colors_suv = torch.stack([u_values, v_values, torch.zeros_like(u_values)], dim=1).float()
                colors_suv = torch.clamp(colors_suv, 0, 1)
            else:
                colors_suv = torch.zeros((0, 3), device='cuda')

            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            
            # Render UV structure
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation_suv, ext, intr, colors_overwrite=colors_suv)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            
            images.append(image)
            
        return torch.stack(images)
       

class SparseUVStructureLatent(SparseUVStructureLatentVisMixin, TransformConditionedMixin, SparseStructureLatent):
    """
    Sparse UV structure latent dataset
    
    Args:
        roots (str): path to the dataset
        ss_latent_model (str): name of the sparse structure latent model
        suv_latent_model (str): name of the UV structure latent model
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        pretrained_ss_dec (str): name of the pretrained sparse structure decoder
        ss_dec_path (str): path to the sparse structure decoder, if given, will override the pretrained_ss_dec
        ss_dec_ckpt (str): name of the sparse structure decoder checkpoint
        pretrained_suv_dec (str): name of the pretrained UV structure decoder
        suv_dec_path (str): path to the UV structure decoder, if given, will override the pretrained_suv_dec
        suv_dec_ckpt (str): name of the UV structure decoder checkpoint
        seperate_pose_latent (bool): whether to seperate pose latent from the ss latent
    """
    def __init__(self,
        roots: str,
        *,
        ss_latent_model: str,
        suv_latent_model: str,
        ss_normalization: Optional[dict] = None,
        pretrained_ss_dec: str = 'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        suv_latent_channels: Optional[int] = None,
        suv_normalization: Optional[dict] = None,
        pretrained_suv_dec: Optional[str] = None,
        suv_dec_path: Optional[str] = None,
        suv_dec_ckpt: Optional[str] = None,
        seperate_pose_latent: bool = False,
        **kwargs
    ):
        self.suv_latent_model = suv_latent_model
        self.suv_normalization = suv_normalization
        self.seperate_pose_latent = seperate_pose_latent

        super().__init__(
            roots,
            latent_model=ss_latent_model,
            normalization=ss_normalization,
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
            pretrained_suv_dec=pretrained_suv_dec,
            suv_dec_path=suv_dec_path,
            suv_dec_ckpt=suv_dec_ckpt,
            suv_latent_channels=suv_latent_channels,
            **kwargs
        )
        
        if self.suv_normalization is not None:
            self.mean = torch.tensor(self.suv_normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.suv_normalization['std']).reshape(-1, 1, 1, 1)
  
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata[f'suv_latent_{self.suv_latent_model}']]
        stats['With UV structure latents'] = len(metadata)
        return metadata, stats

    def _get_suv_z(self, root, instance, frame_data, crop_size_ratio=None):
        view_name = os.path.splitext(frame_data['file_path'])[0]
        suv_latent = np.load(os.path.join(root, 'suv_latents', self.suv_latent_model, f'{instance}.npz'))
        if crop_size_ratio is None:
            suv_z_name = f'{view_name}_mean'
        else:
            suv_z_name = f'{view_name}_mean_crop{int(crop_size_ratio * 1e4)}'
        suv_z = torch.tensor(suv_latent[suv_z_name]).float()
        if self.suv_normalization is not None:
            suv_z = (suv_z - self.mean) / self.std
        return suv_z

    def get_instance(self, root, instance, frame_data, **kwargs):
        pack = super().get_instance(root, instance, frame_data=frame_data, **kwargs)
        if getattr(self, 'crop_size_ratio', None) is None:
            if self.seperate_pose_latent:
                pack['P_0'] = self._get_suv_z(root, instance, frame_data)
                pack['P_0'] = pack['P_0'].unsqueeze(0)
            else:
                pack['x_0'] = torch.cat([pack['x_0'], self._get_suv_z(root, instance, frame_data)], dim=0)
        return pack

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[torch.Tensor, dict]):
        """Visualize both sparse structure and UV structure from concatenated latent."""
        if self.seperate_pose_latent:
            # Only visualize sparse structure
            ss_z = x_0
            return SparseStructureLatent.visualize_sample(self, ss_z)
        else:
            x_0 = x_0 if isinstance(x_0, torch.Tensor) else x_0['x_0']
            # Split the concatenated latent to get ss_z and suv_z
            ss_z = x_0[:, :-self.suv_latent_channels]
            suv_z = x_0[:, -self.suv_latent_channels:]
        
            # Visualize both sparse structure and UV structure
            suv_images = SparseUVStructureLatentVisMixin.visualize_sample(self, suv_z)
            if ss_z.numel() > 0:
                ss_images = SparseStructureLatent.visualize_sample(self, ss_z)
                # Combine images side by side in the X axis
                return torch.cat([ss_images, suv_images], dim=-1)
            else:
                return suv_images


class SparseUVStructureLatentCropMixin:
    def get_instance(self, root, instance, frame_data, **kwargs):
        pack = super().get_instance(root, instance, frame_data=frame_data, **kwargs)
        crop_size_ratio = getattr(self, 'last_crop_ratio', None)
        if crop_size_ratio is not None:
            if self.seperate_pose_latent:
                pack['P_0'] = self._get_suv_z(root, instance, frame_data, crop_size_ratio)
                pack['P_0'] = pack['P_0'].unsqueeze(0)
            else:
                pack['x_0'] = torch.cat([pack['x_0'], self._get_suv_z(root, instance, frame_data, crop_size_ratio)], dim=0)
        return pack


class ImageConditionedSparseUVStructureLatent(SparseUVStructureLatentCropMixin, ImageConditionedMixin, SparseUVStructureLatent):
    """
    Image-conditioned sparse UV structure dataset
    """
    pass


class SparseUVStructureWithLatent(SparseUVStructureLatent, SparseUVStructure):
    """
    Sparse UV structure dataset with sparse UV structure latent (mainly for evaluation)
    """
    @torch.no_grad()
    def visualize_sample(self, sample: Union[torch.Tensor, dict]):
        return SparseUVStructure.visualize_sample(self, sample)


class ImageConditionedSparseUVStructureWithLatent(SparseUVStructureLatentCropMixin, ImageConditionedMixin, SparseUVStructureWithLatent):
    """
    Image-conditioned sparse UV structure dataset with sparse UV structure latent
    """
    pass