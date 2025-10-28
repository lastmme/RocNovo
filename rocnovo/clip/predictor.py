import logging
from typing import Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import pandas as pd

from ..tokenizer.peptide import PTMPeptideTokenizer
from ..common.config import CLIPPredictorConfig, Checkpoint
from .clip import CLIP, CLIPFinetune
from ..numba_utils import get_candidate_ptm_indices, cosine_similarity

@dataclass
class IdentificationMetaData:
    spectra_embedding: npt.NDArray
    spectra_precursor_mz: npt.NDArray
    charges: npt.NDArray

    sequences: npt.NDArray
    sequences_embedding: npt.NDArray
    sequences_mass: npt.NDArray

logger = logging.getLogger(__name__)


class CLIPPredictor:
    def __init__(
        self,
        gpu_id: int,  # 类似于 device
        model: Union[CLIP, CLIPFinetune],
        config: CLIPPredictorConfig,
        tokenizer: PTMPeptideTokenizer,
        test_dataloader: DataLoader,
    ) -> None:
        """Initialize a ModelRunner"""
        self.model = model
        self.model = self.model.to(gpu_id)
        self.test_dataloader = test_dataloader
        self.tokenizer = tokenizer
        self.is_show_progress_bar = config.show_progress_bar
        self.gpu_id = gpu_id
    
    def cal_cosine_score(self, logits_a: torch.Tensor, logits_b: torch.Tensor):
        logits_a_norm = F.normalize(logits_a, dim=-1)
        logits_b_norm = F.normalize(logits_b, dim=-1)
        cosine_score = torch.matmul(logits_a_norm, logits_b_norm.T)
        return cosine_score

    def top1_indices(self, score: torch.Tensor):
        top1_scores, top1_indices = torch.topk(score, k=1, dim=-1)
        return top1_indices.squeeze(1), top1_scores.squeeze(1)

    def unpack_batch(self, batch):
        batch = [item.to(self.gpu_id) for item in batch]
        spectra, precursors, spectra_mask, tokens, tokens_mask = batch
        return spectra, precursors, spectra_mask, tokens, tokens_mask

    def _run_step(self, spectra, precursors, spectra_mask, tokens, tokens_mask) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_per_spec, logits_per_tgt = self.model(
            spectra=spectra,
            precursors=precursors,
            tokens=tokens,
            spectra_mask=spectra_mask,
            tokens_mask=tokens_mask,
        )
        return logits_per_spec, logits_per_tgt

    def pre_iden_metadata(self):
        self.model.eval()

        loop = self.test_dataloader
        if self.is_show_progress_bar:
            loop = tqdm(loop, desc="Preparing Metadata")
        
        metadata = {
            "spectra_embedding": [],
            "spectra_precursor_mz": [],
            "charges": [],
            "sequences": [],
            "sequences_mass": [],
            "sequences_embedding": []
        }

        with torch.no_grad():
            for batch in loop:
                spectra, precursors, spectra_mask, tokens, tokens_mask = self.unpack_batch(batch)
                logits_per_spec, logits_per_tgt = self._run_step(
                    spectra,
                    precursors,
                    spectra_mask,
                    tokens,
                    tokens_mask
                )
                precursors = precursors.cpu().numpy()
                tokens = tokens.cpu().numpy()
                metadata["spectra_embedding"].append(logits_per_spec.cpu().numpy())
                metadata["spectra_precursor_mz"].append(precursors[:, 2])
                metadata["charges"].append(precursors[:, 1])
                metadata["sequences_embedding"].append(logits_per_tgt.cpu().numpy())
                
                sequences = self.tokenizer.detokenize_by_array(tokens)
                mass_seq = self.tokenizer.array2mass(tokens)

                metadata["sequences"].append(sequences)
                metadata["sequences_mass"].append(mass_seq)
        
        for k, v in metadata.items():
            metadata[k] = np.concatenate(v)
    
        return IdentificationMetaData(**metadata)

    def precursor_iden(self, ppm: int, output_path: str):
        metadata = self.pre_iden_metadata()
        
        loop = zip(metadata.charges, metadata.spectra_precursor_mz, metadata.spectra_embedding)
        if self.is_show_progress_bar:
            loop = tqdm(loop, total=metadata.charges.shape[0], desc="Identification")
        
        outputs = []
        idx = 0 # TODO: 改为 scan 号

        for charge, precursor_mz, spectrum_embedding in loop:
            indices, ppm_error, seq_mz = get_candidate_ptm_indices(metadata.sequences_mass, charge, precursor_mz, ppm)
            if len(indices) == 0:
                outputs.append([
                    idx,
                    int(charge),
                    precursor_mz,
                    0,
                    0,
                    "",
                    "",
                    0,
                    False,
                    False
                ])
                idx += 1
                continue

            candidate_sequences = metadata.sequences[indices]
            a = np.array([spectrum_embedding])
            b = metadata.sequences_embedding[indices]
            cosine_score = cosine_similarity(a, b)
            top1_index = np.argmax(cosine_score, axis=1)[0]
            predicted_seq = candidate_sequences[top1_index]
            target_seq = metadata.sequences[idx]
            outputs.append([
                idx,
                int(charge),
                precursor_mz,
                float(seq_mz[top1_index]),
                float(ppm_error[top1_index]),
                predicted_seq,
                target_seq,
                float(cosine_score[0][top1_index]),
                predicted_seq == target_seq,
                True
            ])
            idx += 1
        
        columns = [
            "idx",
            "charge",
            "spectrum_precursor_mz",
            "precursor_mz",
            "ppm",
            "predicted_sequence",
            "true_sequence",
            "similarity",
            "is_true",
            "is_valid"
        ]
        df = pd.DataFrame(outputs, columns=columns)
        df.to_csv(output_path, index=None)
        logger.info("DONE")

    def _step_predict(self, batch):
        spectra, precursors, spectra_mask, tokens, tokens_mask = self.unpack_batch(batch)
        batch_size = spectra.shape[0]
        logits_per_spec, logits_per_tgt = self._run_step(
            spectra,
            precursors,
            spectra_mask,
            tokens,
            tokens_mask
        )
        cosine_score = self.cal_cosine_score(logits_per_spec, logits_per_tgt)
        top1_indices, top1_scores = self.top1_indices(cosine_score)
        results = []
        target_sequences = self.tokenizer.detokenize_by_array(tokens.cpu().numpy())
        predicted_sequences = self.tokenizer.detokenize_by_array(tokens[top1_indices].cpu().numpy())
        count = 0
        for i in range(batch_size):
            target = target_sequences[i]
            predicted = predicted_sequences[i]
            result = [
                target,                          # target sequence
                predicted,                       # predicted sequence
                precursors[i][2].item(),         # precursor_mz
                int(precursors[i][1].item()),  # charge
                top1_scores[i].item(),           # cosine score
                predicted == target              # flag
            ]
            results.append(result)
            count += 1 if target == predicted else 0

        return results, count / batch_size

    def predict(self, output_path: str):
        self.model.eval()

        loop = self.test_dataloader
        if self.is_show_progress_bar:
            loop = tqdm(loop, desc="Prediction")

        end_idx = 0
        idx_seq = []
        results = []
        with torch.no_grad():
            for batch in loop:
                outputs, p_acc = self._step_predict(batch)
                results.extend(outputs)
                idx_seq.extend(list(range(end_idx, end_idx + len(outputs))))
                end_idx += len(outputs)
                loop.set_postfix(p_acc=f"{p_acc:.4f}")

        columns = [
            "target",
            "predict",
            "precursor_mz",
            "precursor_charge",
            "similarity",
            "is_true",
        ]
        df = pd.DataFrame(results, columns=columns)
        df["idx"] = idx_seq
        columns = ["idx"] + columns
        df.to_csv(output_path, columns=columns, index=False)
        logger.info("DONE")
