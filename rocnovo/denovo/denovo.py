import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional

from abc import ABC, abstractmethod
from ..clip.clip import CLIP
from ..tokenizer.peptide import PTMPeptideTokenizer
from ..components.encoders import (
    RoPESpectrumEncoder_Flash,
)
from ..components.decoders import BiDirectionalDecoder


class BaseModel(ABC):

    @abstractmethod
    def forward(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        tokens: torch.LongTensor,
        tokens_reverse: torch.LongTensor,
        src_padding_mask: torch.BoolTensor,
        tgt_padding_mask: torch.BoolTensor,
        tgt_reverse_padding_mask: torch.BoolTensor,
        cls_token: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def candidates_scores(
        self,
        memory: torch.Tensor,
        tokens: torch.LongTensor,
        tokens_reverse: torch.LongTensor,
        precursors: torch.Tensor,
        memory_padding_mask: torch.BoolTensor,
        tgt_padding_mask: torch.BoolTensor,
        tgt_reverse_padding_mask: torch.BoolTensor,
        cls_token: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def encode(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        src_padding_mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.BoolTensor, Optional[torch.Tensor], Optional[float]]:
        pass

    @abstractmethod
    def decode(
        self,
        memory: torch.Tensor,
        tokens: torch.LongTensor,
        tokens_reverse: torch.LongTensor,
        precursors: torch.Tensor,
        memory_padding_mask: torch.BoolTensor,
        tgt_padding_mask: torch.BoolTensor,
        tgt_reverse_padding_mask: torch.BoolTensor,
        cls_token: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class BiDirectRopeClipModel(BaseModel, nn.Module):
    def __init__(
        self,
        clip_model: CLIP,
        clip_model_dim: int,
        model_dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        tokenizer: PTMPeptideTokenizer = PTMPeptideTokenizer("massivekb", reverse=True),
    ):
        super().__init__()
        self.model_dim = model_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.n_vocab = tokenizer.vocab_size
        self.tokenizer = tokenizer

        self.clip_model = clip_model
        self.clip_model.freeze()

        self.clip_embedding = nn.Sequential(
            nn.Linear(clip_model_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, model_dim),
        )

        self.spectrum_encoder = RoPESpectrumEncoder_Flash(
            dim_model=self.model_dim,
            dim_feedforward=self.dim_feedforward,
            n_layers=self.n_layers,
            n_head=self.n_heads,
            dropout=self.dropout,
        )

        self.peptide_decoder = BiDirectionalDecoder(
            model_dim=self.model_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            tokenizer=self.tokenizer,
        )

    def encode(self, spectra: torch.Tensor, precursors: torch.Tensor, src_padding_mask: torch.BoolTensor):
        memory, mask = self.spectrum_encoder(
            spectra,
            precursors,
            src_padding_mask,
        )  # (B, L, D)
        # memory = self.layer_norm(memory)  # (B, L, D)
        cls_token = self.clip_embedding(
            self.clip_model.repr_spectrum(spectra=spectra, precursors=precursors, spectra_mask=src_padding_mask)
        )  # (B, clip_model_dim)
        return memory, mask, cls_token, None

    def decode(
        self,
        memory: torch.Tensor,
        tokens: torch.LongTensor,
        tokens_reverse: torch.LongTensor,
        precursors: torch.Tensor,
        memory_padding_mask: torch.BoolTensor,
        tgt_padding_mask: torch.BoolTensor,
        tgt_reverse_padding_mask: torch.BoolTensor,
        cls_token: torch.Tensor = None,
    ):
        logits, logits_reverse = self.peptide_decoder(
            memory=memory,
            tokens=tokens,
            tokens_reverse=tokens_reverse,
            precursors=precursors,
            memory_padding_mask=memory_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
            tgt_reverse_padding_mask=tgt_reverse_padding_mask,
            Mem_token=cls_token,
        )
        return logits, logits_reverse

    def forward(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        tokens: torch.LongTensor,
        tokens_reverse: torch.LongTensor,
        src_padding_mask: torch.BoolTensor,
        tgt_padding_mask: torch.BoolTensor,
        tgt_reverse_padding_mask: torch.BoolTensor,
        cls_token: torch.Tensor = None,
    ) -> tuple[torch.Tensor, Optional[float]]:
        memory, memory_padding_mask, cls_token, _ = self.encode(spectra, precursors, src_padding_mask)
        return self.decode(
            memory,
            tokens,
            tokens_reverse,
            precursors,
            memory_padding_mask,
            tgt_padding_mask,
            tgt_reverse_padding_mask,
            cls_token,
        )

    def candidates_scores(
        self,
        memory: torch.Tensor,
        tokens: torch.LongTensor,
        tokens_reverse: torch.LongTensor,
        precursors: torch.Tensor,
        memory_padding_mask: torch.BoolTensor,
        tgt_padding_mask: torch.BoolTensor,
        tgt_reverse_padding_mask: torch.BoolTensor,
        cls_token: torch.Tensor = None,
    ) -> torch.Tensor:
        logits, logits_reverse = self.decode(
            memory,
            tokens,
            tokens_reverse,
            precursors,
            memory_padding_mask,
            tgt_padding_mask,
            tgt_reverse_padding_mask,
            cls_token,
        )
        return torch.log_softmax(logits[:, -1, :], dim=-1), torch.log_softmax(logits_reverse[:, -1, :], dim=-1)

