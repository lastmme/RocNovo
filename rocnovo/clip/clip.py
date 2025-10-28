import torch
import torch.nn as nn
import einops
from ..components.encoders import (
    PositionalEncoder,
    SpectrumEncoder,
    MultiFeedForwardModule,
    RoPESpectrumEncoder_Flash,
)
from ..components.mask import encoder_causal_mask
from ..tokenizer.peptide import PAD, SPECIAL_TOKENS, CANONICAL
from ..components.RoPETransformer import (
    RopeFlashAttnEncoder,
    apply_rotary_pos_emb,
    PeakRotaryPositionalEmbeddings,
)


class PeptideEncoder(torch.nn.Module):

    def __init__(
        self,
        n_vocab: int = len(CANONICAL) + len(SPECIAL_TOKENS),
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0,
    ):
        super().__init__()

        # amino acid embedding
        self.aa_encoder = torch.nn.Embedding(
            n_vocab,
            dim_model,
            padding_idx=SPECIAL_TOKENS[PAD],
        )

        self.pos_encoder = PositionalEncoder(dim_model)

        # Transformer encoder
        layer = torch.nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
            norm=torch.nn.LayerNorm(dim_model),
        )

        self.ln_final = nn.LayerNorm(dim_model)

        self.projection = torch.nn.Linear(dim_model, dim_model, bias=False)

    def forward(self, tokens: torch.LongTensor):
        seq_len = tokens.shape[1]
        encoded = self.aa_encoder(tokens)
        encoded = self.pos_encoder(encoded)
        causal_mask = encoder_causal_mask(seq_len, tokens.device)
        latent = self.transformer_encoder(encoded, mask=causal_mask, is_causal=True)
        latent = self.ln_final(latent)
        latent = self.projection(latent)
        return latent, causal_mask

    @property
    def device(self):
        return next(self.parameters()).device


class RopePeptideEncoder(torch.nn.Module):

    def __init__(
        self,
        n_vocab: int = len(CANONICAL) + len(SPECIAL_TOKENS),
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0,
    ):
        super().__init__()
        self.model_dim = dim_model
        self.n_layers = n_layers
        self.n_heads = n_head
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.n_vocab = n_vocab

        # amino acid embedding
        self.aa_encoder = torch.nn.Embedding(
            n_vocab,
            dim_model,
            padding_idx=SPECIAL_TOKENS[PAD],
        )

        self.d_k = self.model_dim // self.n_heads
        self.rotary_emb = PeakRotaryPositionalEmbeddings(
            dim_model=self.d_k,
            min_wavelength=1,
            max_wavelength=1000,
        )
        # Transformer encoder

        self.transformer_encoder = RopeFlashAttnEncoder(
            dim_model=self.model_dim,
            n_head=self.n_heads,
            n_layers=self.n_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )

        self.projection = torch.nn.Linear(dim_model, dim_model, bias=False)

    def forward(self, tokens: torch.LongTensor):
        seq_len = tokens.shape[1]
        encoded = self.aa_encoder(tokens)
        positions = torch.arange(0, seq_len, device=self.device, dtype=torch.float).unsqueeze(0)  # (1, L)
        positions = einops.repeat(positions, "1 L -> B L", B=tokens.size(0))
        pos_emb = self.rotary_emb(positions)

        causal_mask = encoder_causal_mask(seq_len, tokens.device)
        latent = self.transformer_encoder(encoded, pos_emb, src_mask=causal_mask)
        latent = self.projection(latent)
        return latent, causal_mask

    @property
    def device(self):
        return next(self.parameters()).device


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        # Spectrum:
        Spectrum_heads: int = 8,
        Spectrum_layers: int = 9,
        Spectrum_dropout: float = 0.0,
        Spectrum_feedforward: int = 2048,
        # Peptide:
        n_vocab: int = len(CANONICAL) + len(SPECIAL_TOKENS),
        Peptide_heads: int = 8,
        Peptide_layers: int = 9,
        Peptide_dropout: float = 0.0,
        Peptide_feedforward: int = 2048,
    ):
        super().__init__()

        self.spectrumEncoder = SpectrumEncoder(
            dim_model=embed_dim,
            n_head=Spectrum_heads,
            dim_feedforward=Spectrum_feedforward,
            dropout=Spectrum_dropout,
            n_layers=Spectrum_layers,
        )

        self.peptideEncoder = PeptideEncoder(
            n_vocab=n_vocab,
            dim_model=embed_dim,
            n_head=Peptide_heads,
            n_layers=Peptide_layers,
            dim_feedforward=Peptide_feedforward,
            dropout=Peptide_dropout,
        )

        self.global_spectrum = nn.Linear(embed_dim, 1, bias=False)  # 用于计算 spectrum 的全局表示

    def encode_spectrum(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        spectra_mask: torch.BoolTensor,
    ):
        """如果后续需要整个 encoder 张量，调用该函数即可"""
        pkt, mask = self.spectrumEncoder(spectra, precursors, spectra_mask)  # (B, L + 1, dim)
        return pkt, mask

    def encode_peptide(self, tokens: torch.LongTensor):
        tgt, mask = self.peptideEncoder(tokens)  # (B, L, dim)
        return tgt, mask

    def repr_spectrum(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        spectra_mask: torch.BoolTensor,
    ):
        """如果仅需要质谱的嵌入，调用该函数即可"""
        pkt, mask = self.encode_spectrum(spectra, precursors, spectra_mask)
        # pkt: (B, L + 1, dim), mask: (B, L + 1)
        # 取所有 token 的池化向量作为 spectrum 的表示
        weight_scores = self.global_spectrum(pkt)  # (B, L + 1, 1)
        weight_scores = weight_scores.squeeze(-1)  # (B, L + 1)
        weight_scores = weight_scores.masked_fill(mask, float("-inf"))  # (B, L + 1)
        weights = torch.softmax(weight_scores, dim=1)  # (B, L + 1)

        pkt = torch.matmul(weights.unsqueeze(1), pkt).squeeze(1)  # (B, dim)
        return pkt  # (B, dim)

    def repr_peptide(self, tokens: torch.LongTensor, mask: torch.BoolTensor):
        tgt, _ = self.encode_peptide(tokens)
        last_valid_idx = torch.sum(~mask, dim=1) - 1
        return tgt[torch.arange(tgt.shape[0]), last_valid_idx]  # (B, dim)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        tokens: torch.LongTensor,
        spectra_mask: torch.Tensor,
        tokens_mask: torch.Tensor,
    ):
        pkt_features = self.repr_spectrum(spectra, precursors, spectra_mask)
        tgt_features = self.repr_peptide(tokens, tokens_mask)
        return pkt_features, tgt_features


class CLIPFinetune(nn.Module):
    def __init__(self, clip_model: CLIP, dim_input: int, dim_hidden: int, dim_output: int):
        super().__init__()
        self.clip_model = clip_model
        self.clip_model.freeze()
        self.fnn = MultiFeedForwardModule(dim_input, dim_hidden, dim_output, dropout_last_layer=False)

    def encode_spectrum(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        spectra_mask: torch.BoolTensor,
    ):
        pkt, mask = self.clip_model.encode_spectrum(spectra, precursors, spectra_mask)  # (B, L + 1, dim)
        pkt = self.fnn(pkt)
        return pkt, mask

    def encode_peptide(self, tokens: torch.LongTensor):
        tgt, mask = self.clip_model.encode_peptide(tokens)  # (B, L, dim)
        tgt = self.fnn(tgt)
        return tgt, mask

    def repr_spectrum(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        spectra_mask: torch.BoolTensor,
    ):
        """如果仅需要质谱的嵌入，调用该函数即可"""
        pkt, mask = self.encode_spectrum(spectra, precursors, spectra_mask)
        # pkt: (B, L + 1, dim), mask: (B, L + 1)
        # 取所有 token 的池化向量作为 spectrum 的表示
        weight_scores = self.global_spectrum(pkt)  # (B, L + 1, 1)
        weight_scores = weight_scores.squeeze(-1)  # (B, L + 1)
        weight_scores = weight_scores.masked_fill(mask, float("-inf"))  # (B, L + 1)
        weights = torch.softmax(weight_scores, dim=1)  # (B, L + 1)

        pkt = torch.matmul(weights.unsqueeze(1), pkt).squeeze(1)  # (B, dim)
        return pkt  # (B, dim)

    def repr_peptide(self, tokens: torch.LongTensor, mask: torch.BoolTensor):
        tgt = self.encode_peptide(tokens)
        last_valid_idx = torch.sum(~mask, dim=1) - 1
        return tgt[torch.arange(tgt.shape[0]), last_valid_idx]  # (B, dim)

    def forward(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        tokens: torch.LongTensor,
        spectra_mask: torch.Tensor,
        tokens_mask: torch.Tensor,
    ):
        pkt = self.repr_spectrum(spectra, precursors, spectra_mask)
        tgt = self.repr_peptide(tokens, tokens_mask)
        return pkt, tgt


class CLIP_Rope(CLIP):
    def __init__(
        self,
        embed_dim=512,
        Spectrum_heads=8,
        Spectrum_layers=9,
        Spectrum_dropout=0,
        Spectrum_feedforward=2048,
        n_vocab=len(CANONICAL) + len(SPECIAL_TOKENS),
        Peptide_heads=8,
        Peptide_layers=9,
        Peptide_dropout=0,
        Peptide_feedforward=2048,
    ):
        super().__init__(
            embed_dim,
            Spectrum_heads,
            Spectrum_layers,
            Spectrum_dropout,
            Spectrum_feedforward,
            n_vocab,
            Peptide_heads,
            Peptide_layers,
            Peptide_dropout,
            Peptide_feedforward,
        )
        self.spectrumEncoder = RoPESpectrumEncoder_Flash(
            embed_dim,
            Spectrum_heads,
            Spectrum_feedforward,
            Spectrum_layers,
            Spectrum_dropout,
        )
        self.peptideEncoder = RopePeptideEncoder(
            n_vocab,
            embed_dim,
            Peptide_heads,
            Peptide_feedforward,
            Peptide_layers,
            Peptide_dropout,
        )
