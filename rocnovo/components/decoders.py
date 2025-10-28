import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from .encoders import FloatEncoder
from .RoPETransformer import PeakRotaryPositionalEmbeddings, RopeFlashAttnDecoder
from ..tokenizer.peptide import PTMPeptideTokenizer, SPECIAL_TOKENS, PAD, CANONICAL, MASSIVEKB



class TgtEncoder(nn.Module):
    def __init__(self, dim_model: int, dim_feedforward: int, tokenizer: PTMPeptideTokenizer):
        super().__init__()
        self.model_dim = dim_model
        self.dim_feedforward = dim_feedforward
        self.tokenizer = tokenizer

        # precursor token
        self.precursor_mass_dim = self.model_dim // 2
        self.mass_encoder = FloatEncoder(self.precursor_mass_dim)
        self.charge_encoder = nn.Embedding(num_embeddings=10, embedding_dim=self.model_dim - self.precursor_mass_dim)

        # aa encoder
        self.aa_dim = self.model_dim
        self.aa_encoder = nn.Embedding(
            num_embeddings=self.tokenizer.vocab_size,
            embedding_dim=self.aa_dim,
            padding_idx=self.tokenizer.vocab2idx[PAD],
        )

    def forward(
        self,
        tokens: torch.Tensor,
        precursors: torch.Tensor,
        tgt_padding_mask: torch.BoolTensor = None,
        cls_tokens: torch.Tensor = None,
    ):

        masses = self.mass_encoder(precursors[:, None, 0])  # (B, 1, D/2)
        charges = self.charge_encoder(precursors[:, 1].int() - 1)  # (B, 1, D/2)
        precursors_encoded = torch.cat([masses, charges[:, None, :]], dim=-1)  # (B, 1, D)

        # Amino acid tokens
        aa_encoded = self.aa_encoder(tokens)  # (B, L, D)

        # tgt concatenation
        tgt = aa_encoded
        tgt[:, 0, :] = precursors_encoded.squeeze(dim=1)  # (B,L,D)
        if cls_tokens is not None:
            tgt = torch.cat([cls_tokens[:, None, :], tgt], dim=1)  # (B, L+1, D)

        # mask generation
        batch_size, seq_len, _ = tgt.shape
        tgt_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1).to(self.device)
        
        if cls_tokens is not None and tgt_padding_mask is not None:
            tgt_padding_mask = torch.cat(
                [torch.zeros((batch_size, 1), dtype=torch.bool).to(self.device), tgt_padding_mask], dim=1
            )  # (B, L+1)

        return tgt, tgt_mask, tgt_padding_mask,

    @property
    def device(self):
        return next(self.parameters()).device


class BiDirectionalDecoder(nn.Module):
    def __init__(
        self,
        model_dim: int = 512,
        n_layers: int = 9,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
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

        self.tgt_embed = TgtEncoder(
            dim_model=self.model_dim,
            dim_feedforward=self.dim_feedforward,
            tokenizer=self.tokenizer,
        )

        self.d_k = self.model_dim // self.n_heads
        self.rotary_emb = PeakRotaryPositionalEmbeddings(
            dim_model=self.d_k,
            min_wavelength=1,
            max_wavelength=1000,
        )

        self.head = nn.Linear(model_dim, self.n_vocab)
        self.decoder = RopeFlashAttnDecoder(
            dim_model=self.model_dim,
            n_head=self.n_heads,
            n_layers=self.n_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )

        self.head_reverse = nn.Linear(model_dim, self.n_vocab)
        self.decoder_reverse = RopeFlashAttnDecoder(
            dim_model=self.model_dim,
            n_head=self.n_heads,
            n_layers=self.n_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )

    def forward(
        self,
        memory: torch.Tensor,
        tokens: torch.LongTensor,
        tokens_reverse: torch.LongTensor,
        precursors: torch.Tensor,
        memory_padding_mask: torch.BoolTensor,
        tgt_padding_mask: torch.BoolTensor,
        tgt_reverse_padding_mask: torch.BoolTensor,
        Mem_token: torch.Tensor = None,
    ) -> torch.Tensor:
        tgt, tgt_mask, tgt_padding_mask = self.tgt_embed(tokens, precursors, tgt_padding_mask, Mem_token)
        tgt_reverse, tgt_reverse_mask, tgt_reverse_padding_mask = self.tgt_embed(
            tokens_reverse, precursors, tgt_reverse_padding_mask, Mem_token
        )

        positions = torch.arange(0, tgt.size(1), device=tgt.device, dtype=torch.float).unsqueeze(0)  # (1, L+1)
        positions = einops.repeat(positions, "1 L -> B L", B=tgt.size(0))
        pos_emb = self.rotary_emb(positions)

        latent = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
            tgt_mask=tgt_mask,
            pos_emb=pos_emb,
        )

        logits = self.head(latent)
        
        latent_reverse = self.decoder_reverse(
            tgt=tgt_reverse,
            memory=memory,
            tgt_key_padding_mask=tgt_reverse_padding_mask,
            memory_key_padding_mask = memory_padding_mask,
            tgt_mask=tgt_reverse_mask,
            pos_emb=pos_emb,
        )
        logits_reverse = self.head_reverse(latent_reverse)
        return logits[:, 1:],logits_reverse[:,1:]
