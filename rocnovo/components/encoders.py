"""Simple encoders for input into Transformers and the like."""

import math
from typing import Union, Iterable, Literal

import torch
import torch.nn as nn
import einops
import numpy as np

from ..tokenizer.peptide import PTMPeptideTokenizer, PAD, SOS, H2O, NH3
from .RoPETransformer import  PeakRotaryPositionalEmbeddings, RopeFlashAttnEncoder
from .FlashTransformer import FlashEncoder


class FloatEncoder(torch.nn.Module):
    """Encode floating point values using sine and cosine waves.

    Parameters
    ----------
    dim_model : int
        The number of features to output.
    min_wavelength : float
        The minimum wavelength to use.
    max_wavelength : float
        The maximum wavelength to use.
    """

    def __init__(self, dim_model, min_wavelength=0.001, max_wavelength=10000):
        """Initialize the MassEncoder"""
        super().__init__()

        # Error checking:
        if min_wavelength <= 0:
            raise ValueError("'min_wavelength' must be greater than 0.")

        if max_wavelength <= 0:
            raise ValueError("'max_wavelength' must be greater than 0.")

        # Get dimensions for equations:
        d_sin = math.ceil(dim_model / 2)
        d_cos = dim_model - d_sin

        base = min_wavelength / (2 * np.pi)
        scale = max_wavelength / min_wavelength
        sin_exp = torch.arange(0, d_sin).float() / (d_sin - 1)
        cos_exp = (torch.arange(d_sin, dim_model).float() - d_sin) / (d_cos - 1)
        sin_term = base * (scale**sin_exp)
        cos_term = base * (scale**cos_exp)

        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)

    def forward(self, X):
        """Encode m/z values.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, n_masses)
            The masses to embed.

        Returns
        -------
        torch.Tensor of shape (batch_size, n_masses, dim_model)
            The encoded features for the mass spectra.
        """
        sin_mz = torch.sin(X[:, :, None] / self.sin_term)
        cos_mz = torch.cos(X[:, :, None] / self.cos_term)
        return torch.cat([sin_mz, cos_mz], axis=-1)


class MultiFeedForwardModule(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, Iterable[int]],
        output_size: int,
        *,
        activation: Literal["relu", "selu", "gelu"] = "gelu",
        dropout: float = 0.1,
        dropout_last_layer: bool = True,
    ):
        super(MultiFeedForwardModule, self).__init__()
        if activation == "relu":
            self._activation = nn.ReLU()
        elif activation == "selu":
            self._activation = nn.SELU()
        elif activation == "gelu":
            self._activation = nn.GELU()
        else:
            raise ValueError("activation must be relu or selu")

        if not hasattr(hidden_size, "__iter__"):
            if hidden_size is None:
                hidden_size = [output_size]
            else:
                hidden_size = [hidden_size]

        self._layers = []
        layer_dims = [input_size] + hidden_size + [output_size]

        for i in range(1, len(layer_dims) - 1):
            self._layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i]))
            self._layers.append(self._activation)
            self._layers.append(nn.Dropout(dropout))

        self._layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))

        if dropout_last_layer:
            self._layers.append(nn.Dropout(dropout))
        self._layers = nn.Sequential(*self._layers)

    def forward(self, x):
        return self._layers(x)


class PositionalEncoder(FloatEncoder):
    """The positional encoder for sequences.

    Parameters
    ----------
    dim_model : int
        The number of features to output.
    min_wavelength : float, optional
        The shortest wavelength in the geometric progression.
    max_wavelength : float, optional
        The longest wavelength in the geometric progression.
    """

    def __init__(self, dim_model, min_wavelength=1, max_wavelength=10000):
        """Initialize the MzEncoder"""
        super().__init__(
            dim_model=dim_model,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
        )

    def forward(self, X):
        """Encode positions in a sequence.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, n_sequence, n_features)
            The first dimension should be the batch size (i.e. each is one
            peptide) and the second dimension should be the sequence (i.e.
            each should be an amino acid representation).

        Returns
        -------
        torch.Tensor of shape (batch_size, n_sequence, n_features)
            The encoded features for the mass spectra.
        """
        pos = torch.arange(X.shape[1]).type_as(self.sin_term)
        pos = einops.repeat(pos, "n -> b n", b=X.shape[0])
        sin_in = einops.repeat(pos, "b n -> b n f", f=len(self.sin_term))
        cos_in = einops.repeat(pos, "b n -> b n f", f=len(self.cos_term))

        sin_pos = torch.sin(sin_in / self.sin_term)
        cos_pos = torch.cos(cos_in / self.cos_term)
        encoded = torch.cat([sin_pos, cos_pos], axis=2)
        return encoded + X


class PeakEncoder(nn.Module):
    def __init__(
        self,
        dim_model: int = 128,
        dim_intensity: int = None,
        min_wavelength: float = 0.001,
        max_wavelength: float = 10000,
        learned_intensity_encoding: bool = True,
    ):
        """Initialize the MzEncoder"""
        super().__init__()
        self.dim_model = dim_model
        self.dim_mz = dim_model
        self.learned_intensity_encoding = learned_intensity_encoding
        if dim_intensity is not None:
            if dim_intensity >= dim_model:
                raise ValueError("'dim_intensity' must be less than 'dim_model'")

            self.dim_mz -= dim_intensity
            self.dim_intensity = dim_intensity
        else:
            self.dim_intensity = dim_model

        self.mz_encoder = FloatEncoder(
            dim_model=self.dim_mz,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
        )

        if self.learned_intensity_encoding:
            self.int_encoder = torch.nn.Linear(1, self.dim_intensity, bias=False)
        else:
            self.int_encoder = FloatEncoder(
                dim_model=self.dim_intensity,
                min_wavelength=1e-6,
                max_wavelength=1,
            )

    def forward(self, X):

        m_over_z = X[:, :, 0]
        encoded = self.mz_encoder(m_over_z)

        if self.learned_intensity_encoding:
            int_input = X[:, :, [1]]
        else:
            int_input = X[:, :, 1]

        intensity = self.int_encoder(int_input)
        if self.dim_intensity == self.dim_model:
            return encoded + intensity

        return torch.cat([encoded, intensity], dim=2)


class SpectrumEncoder(torch.nn.Module):
    def __init__(
        self,
        dim_model: int = 128,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.peak_encoder = PeakEncoder(dim_model=dim_model)

        self.precursor_encoder = FloatEncoder(dim_model)
        self.precursor_projection = MultiFeedForwardModule(dim_model, dim_feedforward, dim_model)

        # transformer encoder
        layer = torch.nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
            activation="gelu",
            norm_first=True,  # Use LayerNorm before attention and feedforward
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
            norm=nn.LayerNorm(dim_model),
        )

    def forward(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        spectra_mask: torch.BoolTensor,
    ):
        """
        spectra: (B, L, 2) → peaks
        precursors: (B, 3) → mass_cunsum, precursor_charge, precursor_mz
        """
        device = spectra.device
        batch_size = spectra.shape[0]
        peak_repr = self.peak_encoder(spectra)  # (B, L, dim)

        precursor_mz = precursors[:, [2]]  # (B, 1)
        precursor_repr = self.precursor_encoder(precursor_mz)  # (B, 1, dim)
        precursor_repr = self.precursor_projection(precursor_repr)  # (B, 1, dim)
        precursor_mask = torch.zeros((batch_size, 1), dtype=bool).to(device)  # (B, 1)

        mask = torch.cat([precursor_mask, spectra_mask], dim=1)  # (B, L + 1)
        spectra_repr = torch.cat([precursor_repr, peak_repr], dim=1)  # (B, L + 1, dim)
        spectra_repr = self.transformer_encoder(spectra_repr, src_key_padding_mask=mask)  # (B, L + 1, dim)

        return spectra_repr, mask

    @property
    def device(self):
        return next(self.parameters()).device



class RoPEPeakEncoder(nn.Module):
    def __init__(
        self,
        dim_model,
    ):
        """Initialize the MzEncoder"""
        super().__init__()
        self.dim_model = dim_model
        self.mz_encoder = FloatEncoder(
            dim_model=self.dim_model // 2,
            min_wavelength=0.001,
            max_wavelength=10000,
        )
        self.intensity_encoder = FloatEncoder(
            dim_model=self.dim_model - self.dim_model // 2,
            min_wavelength=1e-6,
            max_wavelength=1,
        )

    def forward(self, X):
        m_over_z = X[:, :, 0]
        encoded_mz = self.mz_encoder(m_over_z)  # (B, L, D/2)
        encoded_intensity = self.intensity_encoder(X[:, :, 1])  # (B, L, D/2)
        encoded = torch.cat([encoded_mz, encoded_intensity], dim=-1)
        return encoded

class RoPESpectrumEncoder_Flash(nn.Module):
    def __init__(
        self,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=9,
        dropout=0.1,
    ):
        super().__init__()

        self.peak_encoder = RoPEPeakEncoder(dim_model=dim_model)

        self.precursor_encoder = FloatEncoder(dim_model)
        self.precursor_projection = MultiFeedForwardModule(dim_model, dim_feedforward, dim_model)

        # transformer encoder
        self.transformer_encoder = RopeFlashAttnEncoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.d_k = dim_model // n_head
        self.rotary_emb = PeakRotaryPositionalEmbeddings(
            dim_model=self.d_k,
            min_wavelength=0.001,
            max_wavelength=10000,
        )

    def forward(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        spectra_mask: torch.BoolTensor,
    ):
        """
        spectra: (B, L, 2) → peaks
        precursors: (B, 3) → mass_cunsum, precursor_charge, precursor_mz
        """
        device = spectra.device
        batch_size = spectra.shape[0]
        peak_repr = self.peak_encoder(spectra)  # (B, L, dim)

        # precursor representation
        precursor_mz = precursors[:, [2]]  # (B, 1)
        precursor_repr = self.precursor_encoder(precursor_mz)  # (B, 1, dim)
        precursor_repr = self.precursor_projection(precursor_repr)  # (B, 1, dim)
        precursor_mask = torch.zeros((batch_size, 1), dtype=bool).to(device)  # (B, 1)

        # src generation
        mask = torch.cat([precursor_mask, spectra_mask], dim=1)  # (B, L + 1)
        src = torch.cat([precursor_repr, peak_repr], dim=1)  # (B, L + 1, dim)

        # Rotary positional encoding
        mz_positions = spectra[:, :, 0]  # (B, L)
        precursor_positions = torch.zeros(batch_size, 1, device=device)
        positions = torch.cat([precursor_positions, mz_positions], dim=1)  # (B, L + 1)
        pos_emb = self.rotary_emb(positions)  # (B, L + 1, dim)

        spectra_repr = self.transformer_encoder(src, pos_emb=pos_emb, src_key_padding_mask=mask)  # (B, L + 1, dim)

        return spectra_repr, mask