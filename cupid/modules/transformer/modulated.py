from typing import *
import torch
import torch.nn as nn
from ..attention import MultiHeadAttention, WeightedMultiHeadAttention
from ..norm import LayerNorm32
from .blocks import FeedForwardNet


def apply_mod_scale_shift(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, seqlens: List[int] | None) -> torch.Tensor:
    if seqlens is None:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    else:
        assert len(seqlens) == scale.shape[1] == shift.shape[1], \
            f"Seqlen size ({len(seqlens)}) does not match number of modulations ({scale.shape[1]})"
        assert sum(seqlens) == x.shape[1], \
            f"Seqlen sum ({sum(seqlens)}) does not match input sequence length ({x.shape[1]})"
        out = torch.empty_like(x)
        cur_idx = 0
        for i, length in enumerate(seqlens):
            end_idx = cur_idx + length
            out[:, cur_idx:end_idx] = x[:, cur_idx:end_idx] * (1 + scale[:, i:i+1]) + shift[:, i:i+1]
            cur_idx = end_idx
        return out


def apply_mod_gate(x: torch.Tensor, gate: torch.Tensor, seqlens: List[int] | None) -> torch.Tensor:
    if seqlens is None:
        return x * gate.unsqueeze(1)
    else:
        assert len(seqlens) == gate.shape[1], \
            f"Seqlen size ({len(seqlens)}) does not match number of modulations ({gate.shape[1]})"
        assert sum(seqlens) == x.shape[1], \
            f"Seqlen sum ({sum(seqlens)}) does not match input sequence length ({x.shape[1]})"
        out = torch.empty_like(x)
        cur_idx = 0
        for i, length in enumerate(seqlens):
            end_idx = cur_idx + length
            out[:, cur_idx:end_idx] = x[:, cur_idx:end_idx] * gate[:, i:i+1]
            cur_idx = end_idx
        return out


class ModulatedTransformerBlock(nn.Module):
    """
    Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, mod_seqlens: List[int] | None) -> torch.Tensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=-1)
        h = self.norm1(x)
        h = apply_mod_scale_shift(h, scale_msa, shift_msa, mod_seqlens)
        h = self.attn(h)
        h = apply_mod_gate(h, gate_msa, mod_seqlens)
        x = x + h
        h = self.norm2(x)
        h = apply_mod_scale_shift(h, scale_mlp, shift_mlp, mod_seqlens)
        h = self.mlp(h)
        h = apply_mod_gate(h, gate_mlp, mod_seqlens)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, mod_seqlens: List[int] | None = None) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, mod_seqlens, use_reentrant=False)
        else:
            return self._forward(x, mod, mod_seqlens)


class ModulatedTransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, mod_seqlens: List[int] | None):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=-1)
        h = self.norm1(x)
        h = apply_mod_scale_shift(h, scale_msa, shift_msa, mod_seqlens)
        h = self.self_attn(h)
        h = apply_mod_gate(h, gate_msa, mod_seqlens)
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h
        h = self.norm3(x)
        h = apply_mod_scale_shift(h, scale_mlp, shift_mlp, mod_seqlens)
        h = self.mlp(h)
        h = apply_mod_gate(h, gate_mlp, mod_seqlens)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, mod_seqlens: List[int] | None = None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, mod_seqlens, use_reentrant=False)
        else:
            return self._forward(x, mod, context, mod_seqlens)
        

class ModulatedTransformerCrossBlockWithWeightedCond(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm weighted conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = WeightedMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, context_weight: torch.Tensor, mod_seqlens: List[int] | None):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=-1)
        h = self.norm1(x)
        h = apply_mod_scale_shift(h, scale_msa, shift_msa, mod_seqlens)
        h = self.self_attn(h)
        h = apply_mod_gate(h, gate_msa, mod_seqlens)
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context, weight=context_weight)
        x = x + h
        h = self.norm3(x)
        h = apply_mod_scale_shift(h, scale_mlp, shift_mlp, mod_seqlens)
        h = self.mlp(h)
        h = apply_mod_gate(h, gate_mlp, mod_seqlens)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, context_weight: torch.Tensor, mod_seqlens: List[int] | None = None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, context_weight, mod_seqlens, use_reentrant=False)
        else:
            return self._forward(x, mod, context, context_weight, mod_seqlens)
        