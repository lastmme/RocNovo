import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FlashMHA(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
        x_v: torch.Tensor,
        attn_mask: torch.Tensor = None,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        B, L_q, _ = x_q.shape
        _, L_k, _ = x_k.shape
        _, L_v, _ = x_v.shape

        assert L_k == L_v, f"Key and Value sequence lengths must be the same, but got L_k={L_k} and L_v={L_v}"

        # B H L D 输入到 SDPA
        q = self.q_proj(x_q).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_k).view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_v).view(B, L_v, self.num_heads, self.head_dim).transpose(1, 2)

        mask = None
        if padding_mask is not None:
            # padding_mask 从 (B, L_k) 变形为 (B, 1, 1, L_k)
            # 以便广播到 (B, n_head, L_q, L_k) 的所有头和查询维度
            mask = padding_mask.unsqueeze(1).unsqueeze(2)

        if attn_mask is not None:
            # atten_mask 的形状是 (L_q, L_k)，直接使用会被广播到 (B, n_head, L_q, L_k)
            # 如果 padding_mask 也存在，则使用逻辑“或”合并两个掩码
            if mask is None:
                mask = attn_mask
            else:
                # (B, 1, 1, L_k) | (L_q, L_k) -> 广播后合并为 (B, 1, L_q, L_k)
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

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, -1)

        return self.out_proj(attn_output)


class FlashEncoderLayer(nn.Module):
    def __init__(self, dim_model: int, n_head: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = FlashMHA(dim_model, n_head, dropout=dropout)
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
        src_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        x = src
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, x_norm, x_norm, attn_mask=src_mask, padding_mask=src_key_padding_mask)
        x = x + self.dropout1(attn_out)

        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout2(ffn_out)
        return x


class FlashEncoder(nn.Module):
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
            [FlashEncoderLayer(dim_model, n_head, dim_feedforward, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(dim_model)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = src
        for layer in self.layers:
            x = layer(x, src_mask, src_key_padding_mask)
        return self.norm(x)


class FlashDecoderLayer(nn.Module):
    def __init__(self, dim_model: int, n_head: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = FlashMHA(dim_model, n_head, dropout=dropout)
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
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        x = tgt
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, x_norm, x_norm, attn_mask=tgt_mask, padding_mask=tgt_key_padding_mask)
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

class FlashDecoder(nn.Module):
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
            [FlashDecoderLayer(dim_model, n_head, dim_feedforward, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(dim_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
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
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
            )
        return self.norm(x)
