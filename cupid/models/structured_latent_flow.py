from typing import *
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder
from ..modules.norm import LayerNorm32
from ..modules import sparse as sp
from ..modules.sparse.transformer import ModulatedSparseTransformerCrossBlock
from .sparse_structure_flow import TimestepEmbedder
from .sparse_elastic_mixin import SparseTransformerElasticMixin
from .. import models
from ..utils import dist_utils


class SparseResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        
        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels, bias=True),
        )
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        self.updown = None
        if self.downsample:
            self.updown = sp.SparseDownsample(2)
        elif self.upsample:
            self.updown = sp.SparseUpsample(2)

    def _updown(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.updown is not None:
            x = self.updown(x)
        return x

    def forward(self, x: sp.SparseTensor, emb: torch.Tensor) -> sp.SparseTensor:
        emb_out = self.emb_layers(emb).type(x.dtype)
        scale, shift = torch.chunk(emb_out, 2, dim=1)

        x = self._updown(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats)) * (1 + scale) + shift
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)

        return h
    

class SLatFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        num_io_res_blocks: int = 2,
        io_block_channels: List[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.num_io_res_blocks = num_io_res_blocks
        self.io_block_channels = io_block_channels
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.use_skip_connection = use_skip_connection
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32

        if self.io_block_channels is not None:
            assert int(np.log2(patch_size)) == np.log2(patch_size), "Patch size must be a power of 2"
            assert np.log2(patch_size) == len(io_block_channels), "Number of IO ResBlocks must match the number of stages"

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(in_channels, model_channels if io_block_channels is None else io_block_channels[0])
        
        self.input_blocks = nn.ModuleList([])
        if io_block_channels is not None:
            for chs, next_chs in zip(io_block_channels, io_block_channels[1:] + [model_channels]):
                self.input_blocks.extend([
                    SparseResBlock3d(
                        chs,
                        model_channels,
                        out_channels=chs,
                    )
                    for _ in range(num_io_res_blocks-1)
                ])
                self.input_blocks.append(
                    SparseResBlock3d(
                        chs,
                        model_channels,
                        out_channels=next_chs,
                        downsample=True,
                    )
                )
            
        transformer_block_cls = ModulatedSparseTransformerCrossBlock

        self.blocks = nn.ModuleList([
            transformer_block_cls(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        self.out_blocks = nn.ModuleList([])
        if io_block_channels is not None:
            for chs, prev_chs in zip(reversed(io_block_channels), [model_channels] + list(reversed(io_block_channels[1:]))):
                self.out_blocks.append(
                    SparseResBlock3d(
                        prev_chs * 2 if self.use_skip_connection else prev_chs,
                        model_channels,
                        out_channels=chs,
                        upsample=True,
                    )
                )
                self.out_blocks.extend([
                    SparseResBlock3d(
                        chs * 2 if self.use_skip_connection else chs,
                        model_channels,
                        out_channels=chs,
                    )
                    for _ in range(num_io_res_blocks-1)
                ])
            
        self.out_layer = sp.SparseLinear(model_channels if io_block_channels is None else io_block_channels[0], out_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.blocks.apply(convert_module_to_f16)
        self.out_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.blocks.apply(convert_module_to_f32)
        self.out_blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: torch.Tensor, **kwargs) -> sp.SparseTensor:
        h = self.input_layer(x).type(self.dtype)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        cond = cond.type(self.dtype)

        skips = []
        # pack with input blocks
        for block in self.input_blocks:
            h = block(h, t_emb)
            skips.append(h.feats)
        
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(h.coords[:, 1:]).type(self.dtype)
        
        for block in self.blocks:
            h = block(h, t_emb, cond)

        # unpack with output blocks
        for block, skip in zip(self.out_blocks, reversed(skips)):
            if self.use_skip_connection:
                h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
            else:
                h = block(h, t_emb)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h.type(x.dtype))
        return h
    

class ElasticSLatFlowModel(SparseTransformerElasticMixin, SLatFlowModel):
    """
    SLat Flow Model with elastic memory management.
    Used for training with low VRAM.
    """
    pass


class LatentConditioningSLatFlowModel(SLatFlowModel):
    """
    SLatFlowModel with view conditioned latent

    Args:
        cond_latent_channels (int): number of channels for the latent conditioning
        pretrained_slat_enc (str): name of the pretrained structured latent encoder
        slat_enc_path (str): path to the structured latent encoder, if given, will override the pretrained_slat_enc
        slat_enc_ckpt (str): name of the structured latent encoder checkpoint
    """
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        cond_latent_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        num_io_res_blocks: int = 2,
        io_block_channels: List[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        pretrained_slat_enc: str = 'microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16', 
        slat_enc_path: Optional[str] = None, 
        slat_enc_ckpt: Optional[str] = None, 
    ):
        super().__init__(
            resolution,
            in_channels + cond_latent_channels,
            model_channels,
            cond_channels,
            out_channels,
            num_blocks,
            num_heads,
            num_head_channels,
            mlp_ratio,
            patch_size,
            num_io_res_blocks,
            io_block_channels,
            pe_mode,
            use_fp16,
            use_checkpoint,
            use_skip_connection,
            share_mod,
            qk_rms_norm,
            qk_rms_norm_cross,
        )
        self.cond_latent_channels = cond_latent_channels
        self.pretrained_slat_enc = pretrained_slat_enc
        self.slat_enc_path = slat_enc_path
        self.slat_enc_ckpt = slat_enc_ckpt
        self.slat_enc = [None]  # Use a list to avoid register the slat_enc as a sub nn.Module

    def _loading_slat_enc(self):
        if self.slat_enc[0] is not None:
            return
        if self.slat_enc_path is not None:
            cfg = json.load(open(os.path.join(self.slat_enc_path, 'config.json'), 'r'))
            encoder = getattr(models, cfg['models']['encoder']['name'])(**cfg['models']['encoder']['args'])
            ckpt_path = os.path.join(self.slat_enc_path, 'ckpts', f'encoder_{self.slat_enc_ckpt}.pt')
            encoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            encoder = models.from_pretrained(self.pretrained_slat_enc)
        self.slat_enc[0] = encoder.eval().cuda()

    def _delete_slat_enc(self):
        del self.slat_enc[0]
        self.slat_enc = [None]

    @torch.no_grad()
    def encode_latent(self, uvs: sp.SparseTensor, cond: torch.Tensor) -> sp.SparseTensor:
        patchtokens = cond[:, 5:].permute(0, 2, 1).reshape(cond.shape[0], 1024, 37, 37)

        cond_feats = []
        for i, uvs_i in enumerate(uvs.unbind(0)):
            sample_coords = uvs_i.feats[None, None, :, :] * 2 - 1
            cond_voxels = F.grid_sample(
                patchtokens[i:i+1], 
                sample_coords, 
                mode='bilinear', 
                padding_mode='zeros',
                align_corners=False,
            )
            cond_voxels = cond_voxels.squeeze(2).squeeze(0).permute(1, 0)
            cond_feats.append(cond_voxels)
        cond_feats = torch.cat(cond_feats, dim=0)
        cond_feats = sp.SparseTensor(feats=cond_feats, coords=uvs.coords)

        with dist_utils.local_master_first():
            self._loading_slat_enc()
        latent = self.slat_enc[0](cond_feats, sample_posterior=False)

        null_condition_mask = (cond[:, :5] == 0).flatten(start_dim=1).all(dim=1).float()
        latent = latent * (1 - null_condition_mask[:, None])

        return latent

    def forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: torch.Tensor, uvs: sp.SparseTensor, **kwargs) -> sp.SparseTensor:
        latent_cond = self.encode_latent(uvs, cond)
        x = x.replace(torch.cat([x.feats, latent_cond.feats], dim=1))
        h = super().forward(x, t, cond, **kwargs)
        return h


class ElasticLatentConditioningSLatFlowModel(SparseTransformerElasticMixin, LatentConditioningSLatFlowModel):
    """
    SLat Flow Model with elastic memory management and latent conditioning.
    Used for training with low VRAM.
    """
    pass


class VisualLatentConditioningSLatFlowModel(LatentConditioningSLatFlowModel):
    """
    SLat Flow Model with visual latent conditioning.

    Args:
        visual_input_channels (int): number of channels for the visual input. Defaults to 3.
        visual_feat_resolution (int): resolution for the first layer of visual feature. Defaults to 256.
        visual_feat_channels (int): number of channels for each layer of visual feature.
        visual_conv_kernel_size (int): kernel size for the visual convolution. Defaults to 3.
    """
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        cond_latent_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        num_io_res_blocks: int = 2,
        io_block_channels: List[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        visual_input_channels: int = 3, 
        visual_feat_resolution: int = 256, 
        visual_feat_channels: List[int] = [4, 4, 4, 4],
        visual_conv_kernel_size: int = 3,
    ):
        super().__init__(
            resolution,
            in_channels + sum(visual_feat_channels),
            model_channels,
            cond_channels,
            cond_latent_channels,
            out_channels,
            num_blocks,
            num_heads,
            num_head_channels,
            mlp_ratio,
            patch_size,
            num_io_res_blocks,
            io_block_channels,
            pe_mode,
            use_fp16,
            use_checkpoint,
            use_skip_connection,
            share_mod,
            qk_rms_norm,
            qk_rms_norm_cross,
        )
        self.visual_input_channels = visual_input_channels
        self.visual_feat_resolution = visual_feat_resolution
        self.visual_feat_channels = visual_feat_channels
        self.visual_conv_kernel_size = visual_conv_kernel_size

        in_resolution = out_resolution = visual_feat_resolution
        in_channels = visual_input_channels
        self.visual_conv_blocks = nn.ModuleList()
        for out_channels in visual_feat_channels:
            is_downsample = out_resolution < in_resolution
            self.visual_conv_blocks.append(nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=visual_conv_kernel_size,
                padding=visual_conv_kernel_size // 2,
                stride=2 if is_downsample else 1,
            ))
            in_channels = out_channels
            in_resolution = out_resolution
            out_resolution = out_resolution // 2

    def encode_visual_feat(self, uvs: sp.SparseTensor, visual_cond: torch.Tensor) -> sp.SparseTensor:
        visual_feats = []
        visual_cond = F.interpolate(
            visual_cond, 
            size=(self.visual_feat_resolution, self.visual_feat_resolution), 
            mode='bilinear', 
            align_corners=False,
            antialias=True,
        )

        for conv_block in self.visual_conv_blocks:
            visual_cond = conv_block(visual_cond)
            visual_feats.append(visual_cond)

        latents = []
        for i, uvs_i in enumerate(uvs.unbind(0)):
            sample_coords = uvs_i.feats[None, None, :, :] * 2 - 1
            sampled_feats = []
            for visual_feat in visual_feats:
                feat = F.grid_sample(
                    visual_feat[i:i+1],
                    sample_coords,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False,
                )
                feat = feat.squeeze(2).squeeze(0).permute(1, 0)
                sampled_feats.append(feat)
            latents.append(torch.cat(sampled_feats, dim=1))
        latents = torch.cat(latents, dim=0)
        latents = sp.SparseTensor(feats=latents, coords=uvs.coords)
        
        return latents

    def forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: Dict[str, torch.Tensor], uvs: sp.SparseTensor, **kwargs) -> sp.SparseTensor:
        latent_cond = self.encode_latent(uvs, cond['dino_cond'])
        visual_cond = self.encode_visual_feat(uvs, cond['visual_cond'])
        x = x.replace(torch.cat([x.feats, latent_cond.feats, visual_cond.feats], dim=1))
        h = SLatFlowModel.forward(self, x, t, cond['dino_cond'], **kwargs)
        return h


class ElasticVisualLatentConditioningSLatFlowModel(SparseTransformerElasticMixin, VisualLatentConditioningSLatFlowModel):
    """
    SLat Flow Model with elastic memory management and visual latent conditioning.
    Used for training with low VRAM.
    """
    pass


class PositionalEmbeddingConditioningSLatFlowModel(SLatFlowModel):
    """
    SLatFlowModel with positional embedding conditioning

    Args:
        cond_latent_channels (int): number of channels for the latent conditioning
        cond_pos_embed_channels (int): number of channels for the positional embedding conditioning
        image_cond_model (str): The image conditioning model
    """
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        cond_latent_channels: int,
        cond_pos_embed_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        num_io_res_blocks: int = 2,
        io_block_channels: List[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        image_cond_model: str = 'dinov2_vitl14_reg', 
    ):
        super().__init__(
            resolution,
            in_channels + cond_latent_channels,
            model_channels,
            cond_channels,
            out_channels,
            num_blocks,
            num_heads,
            num_head_channels,
            mlp_ratio,
            patch_size,
            num_io_res_blocks,
            io_block_channels,
            pe_mode,
            use_fp16,
            use_checkpoint,
            use_skip_connection,
            share_mod,
            qk_rms_norm,
            qk_rms_norm_cross,
        )
        self.cond_latent_channels = cond_latent_channels
        self.cond_pos_embed_channels = cond_pos_embed_channels
        self.image_cond_model_name = image_cond_model

        with dist_utils.local_master_first():
            dinov2_model = torch.hub.load('facebookresearch/dinov2', self.image_cond_model_name, pretrained=True)
        dinov2_model = dinov2_model.eval().cuda()
        dinov2_pos_embed = dinov2_model.pos_embed[0, 1:]
        num_tokens_y = num_tokens_x = int(np.sqrt(dinov2_pos_embed.shape[0]))
        dinov2_pos_embed = dinov2_pos_embed.T.reshape(1, dinov2_pos_embed.shape[1], num_tokens_y, num_tokens_x)
        self.register_buffer("cond_pos_embed", dinov2_pos_embed, persistent=False)

        self.pos_embed_proj = nn.Conv2d(cond_pos_embed_channels, cond_latent_channels, 1)

    def encode_latent(self, uvs: sp.SparseTensor) -> sp.SparseTensor:
        pos_embed_latent = self.pos_embed_proj(self.cond_pos_embed)

        cond_latent = []
        for i, uvs_i in enumerate(uvs.unbind(0)):
            sample_coords = uvs_i.feats[None, None, :, :] * 2 - 1
            cond_voxels = F.grid_sample(
                pos_embed_latent, 
                sample_coords, 
                mode='bilinear', 
                padding_mode='zeros',
                align_corners=False,
            )
            cond_voxels = cond_voxels.squeeze(2).squeeze(0).permute(1, 0)
            cond_latent.append(cond_voxels)
        cond_latent = torch.cat(cond_latent, dim=0)
        cond_latent = sp.SparseTensor(feats=cond_latent, coords=uvs.coords)

        return cond_latent

    def forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: torch.Tensor, uvs: sp.SparseTensor, **kwargs) -> sp.SparseTensor:
        latent_cond = self.encode_latent(uvs)
        x = x.replace(torch.cat([x.feats, latent_cond.feats], dim=1))
        h = super().forward(x, t, cond, **kwargs)
        return h


class ElasticPositionalEmbeddingConditioningSLatFlowModel(SparseTransformerElasticMixin, PositionalEmbeddingConditioningSLatFlowModel):
    """
    SLat Flow Model with elastic memory management and positional embedding conditioning.
    Used for training with low VRAM.
    """
    pass


class SpatialConditioningSLatFlowModel(SLatFlowModel):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        cond_latent_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        num_io_res_blocks: int = 2,
        io_block_channels: List[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__(
            resolution,
            in_channels + cond_latent_channels,
            model_channels,
            cond_channels,
            out_channels,
            num_blocks,
            num_heads,
            num_head_channels,
            mlp_ratio,
            patch_size,
            num_io_res_blocks,
            io_block_channels,
            pe_mode,
            use_fp16,
            use_checkpoint,
            use_skip_connection,
            share_mod,
            qk_rms_norm,
            qk_rms_norm_cross,
        )
        self.cond_latent_channels = cond_latent_channels

        self.cond_embedder = nn.Sequential(
            nn.Conv2d(cond_channels, cond_latent_channels * 4, 1),
            nn.SiLU(),
            nn.Conv2d(cond_latent_channels * 4, cond_latent_channels, 1),
        )

    def forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: torch.Tensor, uvs: sp.SparseTensor, **kwargs) -> sp.SparseTensor:
        cond_tokens = cond[:, 5:, :].reshape(-1, 37, 37, cond.shape[-1]).permute(0, 3, 1, 2)
        cond_embeds = self.cond_embedder(cond_tokens)

        cond_feats = []
        for i, uvs_i in enumerate(uvs.unbind(0)):
            sample_coords = uvs_i.feats[None, None, :, :] * 2 - 1
            cond_voxels = F.grid_sample(cond_embeds[i:i+1], sample_coords, mode='bilinear', align_corners=False)
            cond_voxels = cond_voxels.squeeze(2).squeeze(0).permute(1, 0)
            cond_feats.append(cond_voxels)
        cond_feats = torch.cat(cond_feats, dim=0)

        x = x.replace(torch.cat([x.feats, cond_feats], dim=1))
        h = super().forward(x, t, cond, **kwargs)
        return h


class ElasticSpatialConditioningSLatFlowModel(SparseTransformerElasticMixin, SpatialConditioningSLatFlowModel):
    """
    SLat Flow Model with elastic memory management and spatial conditioning.
    Used for training with low VRAM.
    """
    pass