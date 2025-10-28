"""Data loaders for the de novo sequencing task."""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .datasets import DeNovoStream, SpectrumStream, BiDirectDeNovoStream
from ..tokenizer.peptide import SPECIAL_TOKENS, PAD, PROTON


def prepare_spectra(spectra, precursor_mzs, precursor_charges):
    spectra = pad_sequence(spectra, batch_first=True)
    mask = ~torch.sum(spectra, dim=-1).bool()
    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - PROTON) * precursor_charges
    precursors = torch.vstack([precursor_masses, precursor_charges, precursor_mzs]).T.float()
    return spectra, precursors, mask


def spectra_embedding_collate_fn(batch: List[Tuple[torch.Tensor, float, int]]):
    spectra, precursor_mzs, precursor_charges = list(zip(*batch))
    spectra, precursors, mask = prepare_spectra(spectra, precursor_mzs, precursor_charges)
    return spectra, precursors, mask


def denovo_collate_fn(
    batch: List[Tuple[torch.Tensor, float, int, torch.LongTensor]],
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    spectra, precursor_mzs, precursor_charges, peptide_tokens = list(zip(*batch))
    spectra, precursors, spectra_mask = prepare_spectra(spectra, precursor_mzs, precursor_charges)
    peptide_tokens = pad_sequence(peptide_tokens, batch_first=True, padding_value=SPECIAL_TOKENS[PAD])
    peptide_mask = peptide_tokens == SPECIAL_TOKENS[PAD]
    return spectra, precursors, spectra_mask, peptide_tokens, peptide_mask


def bidirect_denovo_collate_fn(
    batch: List[Tuple[torch.Tensor, float, int, torch.LongTensor, torch.LongTensor]],
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    spectra, precursor_mzs, precursor_charges, peptide_tokens, peptide_tokens_reverse = list(zip(*batch))
    spectra, precursors, spectra_mask = prepare_spectra(spectra, precursor_mzs, precursor_charges)
    peptide_tokens = pad_sequence(peptide_tokens, batch_first=True, padding_value=SPECIAL_TOKENS[PAD])
    peptide_tokens_reverse = pad_sequence(peptide_tokens_reverse, batch_first=True, padding_value=SPECIAL_TOKENS[PAD])
    peptide_mask = peptide_tokens == SPECIAL_TOKENS[PAD]
    peptide_reverse_mask = peptide_tokens_reverse == SPECIAL_TOKENS[PAD]
    return spectra, precursors, spectra_mask, peptide_tokens, peptide_mask, peptide_tokens_reverse, peptide_reverse_mask


class SpectraDataLoaderModule:
    def __init__(
        self,
        train_batch_size: int = 128,
        eval_batch_size: int = 128,
        n_workers: Optional[int] = None,
    ):
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.n_workers = n_workers if n_workers is not None else os.cpu_count()

    def train_dataloader(self, train_dataset: SpectrumStream):
        return DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            collate_fn=spectra_embedding_collate_fn,
            num_workers=self.n_workers,
            shuffle=False,
            pin_memory=True,
            sampler=DistributedSampler(train_dataset),
        )

    def val_dataloader(self, valid_dataset: SpectrumStream):
        return DataLoader(
            valid_dataset,
            batch_size=self.train_batch_size,
            collate_fn=spectra_embedding_collate_fn,
            num_workers=self.n_workers,
            shuffle=False,
            pin_memory=True,
            sampler=DistributedSampler(valid_dataset),  # DDP
        )

    def test_dataloader(self, test_dataset: SpectrumStream):
        return DataLoader(
            test_dataset,
            batch_size=self.eval_batch_size,
            collate_fn=spectra_embedding_collate_fn,
            num_workers=self.n_workers,
            shuffle=False,
        )


class DeNovoDataLoaderModule:
    def __init__(
        self,
        train_batch_size: int = 128,
        eval_batch_size: int = 128,
        n_workers: Optional[int] = None,
    ):
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.n_workers = n_workers if n_workers is not None else os.cpu_count()

    def train_dataloader(self, train_dataset: DeNovoStream):
        return DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            collate_fn=denovo_collate_fn,
            num_workers=self.n_workers,
            shuffle=False,  # DDP 模式下, dataloader 中的 shuffle 必须设置为 False
            pin_memory=True,
            sampler=DistributedSampler(train_dataset),  # DDP mode
        )

    def val_dataloader(self, valid_dataset: DeNovoStream):
        return DataLoader(
            valid_dataset,
            batch_size=self.train_batch_size,
            collate_fn=denovo_collate_fn,
            num_workers=self.n_workers,
            shuffle=False,
            pin_memory=True,
            sampler=DistributedSampler(valid_dataset),  # DDP mode
        )

    def test_dataloader(self, test_dataset: DeNovoStream):
        return DataLoader(
            test_dataset,
            batch_size=self.eval_batch_size,
            collate_fn=denovo_collate_fn,
            num_workers=self.n_workers,
            shuffle=False,
        )


class BiDirectDeNovoDataLoaderModule:
    def __init__(
        self,
        train_batch_size: int = 128,
        eval_batch_size: int = 128,
        n_workers: Optional[int] = None,
    ):
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.n_workers = n_workers if n_workers is not None else os.cpu_count()

    def train_dataloader(self, train_dataset: BiDirectDeNovoStream):
        return DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            collate_fn=bidirect_denovo_collate_fn,
            num_workers=self.n_workers,
            shuffle=False,  # DDP 模式下, dataloader 中的 shuffle 必须设置为 False
            pin_memory=True,
            sampler=DistributedSampler(train_dataset),  # DDP mode
        )

    def val_dataloader(self, valid_dataset: BiDirectDeNovoStream):
        return DataLoader(
            valid_dataset,
            batch_size=self.train_batch_size,
            collate_fn=bidirect_denovo_collate_fn,
            num_workers=self.n_workers,
            shuffle=False,
            pin_memory=True,
            sampler=DistributedSampler(valid_dataset),  # DDP mode
        )

    def test_dataloader(self, test_dataset: BiDirectDeNovoStream):
        return DataLoader(
            test_dataset,
            batch_size=self.eval_batch_size,
            collate_fn=bidirect_denovo_collate_fn,
            num_workers=self.n_workers,
            shuffle=False,
        )
