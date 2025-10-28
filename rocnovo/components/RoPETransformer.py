"""
一个完整的谱图编码器实现，它使用了一个可定制的旋转位置编码 (RoPE)，
该 RoPE 允许通过 min_wavelength 和 max_wavelength 来精确控制频率范围。
"""

from typing import Tuple, Union, Iterable, Literal
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .FlashTransformer import FlashMHA



class PeakRotaryPositionalEmbeddings(nn.Module):
    """
    一个可定制的 RoPE 版本，它预先计算用于旋转的 sin 和 cos 值。
    """

    def __init__(self, dim_model: int, min_wavelength: float, max_wavelength: float):
        super().__init__()
        if dim_model % 2 != 0:
            raise ValueError("For RoPE, dim_model must be even.")
        if min_wavelength <= 0 or max_wavelength <= 0 or min_wavelength >= max_wavelength:
            raise ValueError("Wavelengths must be positive and min_wavelength < max_wavelength.")

        freq_min = 1.0 / max_wavelength
        freq_max = 1.0 / min_wavelength

        num_freqs = dim_model // 2
        exponent = torch.arange(num_freqs, dtype=torch.float)
        if num_freqs > 1:
            exponent /= num_freqs - 1

        scale = freq_max / freq_min
        inv_freq = freq_min * (scale**exponent)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预计算 sin 和 cos 编码。
        - positions: (B, L)
        - 返回: (cos_emb, sin_emb), 每个形状为 (B, L, D/2)
        """
        # freqs: (B, L, D/2)
        freqs = torch.einsum("bl,d->bld", positions, self.inv_freq)
        # 直接在 freqs 上计算，使其维度 (D/2) 与 x_real/x_imag 匹配
        return (freqs.cos(), freqs.sin())


def apply_rotary_pos_emb(x: torch.Tensor, pos_emb: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """将预计算的 RoPE 应用于输入张量 x。"""
    cos_emb, sin_emb = pos_emb
    x_real, x_imag = x.chunk(2, dim=-1)
    rotated_real = x_real * cos_emb - x_imag * sin_emb
    rotated_imag = x_real * sin_emb + x_imag * cos_emb
    return torch.cat((rotated_real, rotated_imag), dim=-1)


class MultiHeadAttentionWithRoPE(nn.Module):
    """
    使用预计算 RoPE 的多头注意力机制。
    """

    def __init__(self, dim_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = dim_model // n_head
        self.scale = self.d_k**-0.5

        self.q_proj = nn.Linear(dim_model, dim_model)
        self.k_proj = nn.Linear(dim_model, dim_model)
        self.v_proj = nn.Linear(dim_model, dim_model)
        self.out_proj = nn.Linear(dim_model, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: Tuple[torch.Tensor, torch.Tensor],
        attn_mask: torch.Tensor = None,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)

        # 调整 pos_emb 形状以匹配多头维度
        # (B, L, D_k) -> (B, 1, L, D_k) for broadcasting over heads
        cos_emb = pos_emb[0].unsqueeze(1)
        sin_emb = pos_emb[1].unsqueeze(1)

        q = apply_rotary_pos_emb(q, (cos_emb, sin_emb))
        k = apply_rotary_pos_emb(k, (cos_emb, sin_emb))

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale

        mask = None
        if padding_mask is not None:
            # padding_mask 从 (B, L) 变形为 (B, 1, 1, L)
            # 以便广播到 (B, n_head, L, L) 的所有头和查询维度
            mask = padding_mask.unsqueeze(1).unsqueeze(2)

        if attn_mask is not None:
            # atten_mask 的形状是 (L, L)，会被广播到 (B, n_head, L, L)
            # 如果 padding_mask 也存在，则使用逻辑“或”合并两个掩码
            if mask is None:
                mask = attn_mask
            else:
                # (B, 1, 1, L) | (L, L) -> 广播后合并为 (B, 1, L, L)
                mask = mask | attn_mask
        # 应用 mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, -1e9)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(attn_output)


class FlashRoPEMHA(nn.Module):
    """
    使用 SDPA (Flash Attention) 和 RoPE 的多头注意力。
    """

    def __init__(self, dim_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = dim_model // n_head

        self.q_proj = nn.Linear(dim_model, dim_model)
        self.k_proj = nn.Linear(dim_model, dim_model)
        self.v_proj = nn.Linear(dim_model, dim_model)
        self.out_proj = nn.Linear(dim_model, dim_model)
        self.dropout_p = dropout

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: Tuple[torch.Tensor, torch.Tensor],
        attn_mask: torch.Tensor = None,  # (L,L)
        padding_mask: torch.Tensor = None,  # (B, L)
    ) -> torch.Tensor:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)

        cos_emb = pos_emb[0].unsqueeze(1)
        sin_emb = pos_emb[1].unsqueeze(1)
        q = apply_rotary_pos_emb(q, (cos_emb, sin_emb))
        k = apply_rotary_pos_emb(k, (cos_emb, sin_emb))

        # 最终掩码的形状需要能广播到 (B, H, L, L)。
        mask = None
        if padding_mask is not None:
            # padding_mask 从 (B, L) 变形为 (B, 1, 1, L)
            # 以便广播到 (B, n_head, L, L) 的所有头和查询维度
            mask = padding_mask.unsqueeze(1).unsqueeze(2)

        if attn_mask is not None:
            # atten_mask 的形状是 (L, L)，会被广播到 (B, n_head, L, L)
            # 如果 padding_mask 也存在，则使用逻辑“或”合并两个掩码
            if mask is None:
                mask = attn_mask
            else:
                # (B, 1, 1, L) | (L, L) -> 广播后合并为 (B, 1, L, L)
                mask = mask | attn_mask
        if mask is not None:
            mask = mask.logical_not()
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(attn_output)


class RopeFlashAttnEncoderLayer(nn.Module):
    """使用 Flash attention"""

    def __init__(self, dim_model: int, n_head: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = FlashRoPEMHA(dim_model, n_head, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_model),
        )
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        pos_emb: Tuple[torch.Tensor, torch.Tensor],
        src_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        x = src
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, pos_emb, attn_mask=src_mask, padding_mask=src_key_padding_mask)
        x = x + self.dropout1(attn_out)

        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout2(ffn_out)
        return x


class RopeFlashAttnEncoder(nn.Module):

    def __init__(
        self,
        dim_model: int,
        n_head: int,
        dim_feedforward: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [RopeFlashAttnEncoderLayer(dim_model, n_head, dim_feedforward, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(dim_model)

    def forward(
        self,
        src: torch.Tensor,
        pos_emb: Tuple[torch.Tensor, torch.Tensor],
        src_key_padding_mask: torch.Tensor = None,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = src
        for layer in self.layers:
            x = layer(x, pos_emb, src_mask, src_key_padding_mask)
        return self.norm(x)


class RopeFlashAttnDecoderLayer(nn.Module):
    def __init__(self, dim_model: int, n_head: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = FlashRoPEMHA(dim_model, n_head, dropout=dropout)
        self.cross_attn = FlashMHA(embed_dim=dim_model, num_heads=n_head, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_model),
        )
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        pos_emb: Tuple[torch.Tensor, torch.Tensor],
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        x = tgt
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, pos_emb, attn_mask=tgt_mask, padding_mask=tgt_key_padding_mask)
        x = x + self.dropout1(attn_out)

        x_norm = self.norm2(x)
        cross_attn_out = self.cross_attn(
            x_norm,
            memory,
            memory,
            padding_mask=memory_key_padding_mask,
            attn_mask=memory_mask,
        )
        x = x + self.dropout2(cross_attn_out)

        x_norm = self.norm3(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout3(ffn_out)

        return x


class RopeFlashAttnDecoder(nn.Module):
    """使用 RoPE 的 Transformer 解码器。"""

    def __init__(
        self,
        dim_model: int,
        n_head: int,
        dim_feedforward: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [RopeFlashAttnDecoderLayer(dim_model, n_head, dim_feedforward, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(dim_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        pos_emb: Tuple[torch.Tensor, torch.Tensor],
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = tgt
        for layer in self.layers:
            x = layer(
                x,
                memory,
                pos_emb,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
            )
        return self.norm(x)
