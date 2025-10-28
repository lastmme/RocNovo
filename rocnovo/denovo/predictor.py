import logging
from tqdm import tqdm
import pandas as pd
from dataclasses import asdict
from typing import Union
import sys

import torch
from torch.utils.data import DataLoader

from ..tokenizer.peptide import PTMPeptideTokenizer
from ..common.config import DecoderPredictorConfig, Checkpoint, BeamsearchConfig
from .beamsearch import BeamsearchPredictor, GreedysearchPredictor
from .denovo import BaseModel

logger = logging.getLogger(__name__)


class DenovoPredictor:
    def __init__(
        self,
        gpu_id: int,
        model: BaseModel,
        config: DecoderPredictorConfig,
        tokenizer: PTMPeptideTokenizer,
        test_dataloader: DataLoader,
        checkpoint_path: str,
        output_path: str,
    ) -> None:
        self.model = model
        self.test_dataloader = test_dataloader
        self.tokenizer = tokenizer
        self.is_show_progress_bar = config.show_progress_bar
        self.BeamsearchConfig = BeamsearchConfig(**config.beam_search)
        self.BeamsearchConfig.n_vocab = self.tokenizer.vocab_size
        self.gpu_id = gpu_id
        self.output_path = output_path
        self.checkpoint_path = checkpoint_path
        self.load_from_checkpoint(checkpoint_path)

        if self.BeamsearchConfig.beam_size == 0:
            self.predictor = GreedysearchPredictor(
                model=self.model, tokenizer=self.tokenizer, **asdict(self.BeamsearchConfig)
            )
        else:
            self.predictor = BeamsearchPredictor(
                model=self.model, tokenizer=self.tokenizer, **asdict(self.BeamsearchConfig)
            )
        self.results = []
        self.idx_seq = []

    def load_from_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint = Checkpoint(**checkpoint)
        self.model.load_state_dict(checkpoint.model_state_dict)
        self.model = self.model.to(self.gpu_id)
        logger.info(f"Loaded checkpoint from {path}")

    def unpack_batch(self, batch):
        batch = [item.to(self.gpu_id) for item in batch]
        if len(batch) == 7:
            spectra, precursors, spectra_mask, _, _, _, _ = batch
        elif len(batch) == 3:
            spectra, precursors, spectra_mask = batch
        return spectra, precursors, spectra_mask

    def save_results(self):
        if self.results:
            columns = [
                "precursor_mz",
                "precursor_charge",
                "seq1_score",
                "seq1",
                "seq2_score",
                "seq2",
                "pred_mass",
                "pred_score",
                "pred_peptide",
            ]
            df = pd.DataFrame(self.results, columns=columns)
            df["idx"] = self.idx_seq
            df.to_csv(self.output_path, index=False)
            logger.info(f"Saved partial results to {self.output_path}")
        else:
            logger.info("No results to save.")

    def _step_prediction(self, batch):
        spectra, precursors, spectra_mask = self.unpack_batch(batch)
        batch_size = spectra.shape[0]
        pred_tokens, pred_scores, pred_tokens_reverse, pred_scores_reverse = self.predictor.decode(
            spectra=spectra, precursors=precursors, src_padding_mask=spectra_mask
        )
        results = []
        pred_sequences = self.tokenizer.detokenize_by_array(pred_tokens.cpu().numpy())
        pred_scores = pred_scores.cpu().numpy()
        pred_sequences_reverse = self.tokenizer.reverse_detokenize_by_array(pred_tokens_reverse.cpu().numpy())
        pred_scores_reverse = pred_scores_reverse.cpu().numpy()
        for i in range(batch_size):
            pred_score = pred_scores[i]
            pred_peptide = "" if pred_score == -2.0 else pred_sequences[i]
            pred_score_reverse = pred_scores_reverse[i]
            pred_peptide_reverse = "" if pred_score_reverse == -2.0 else pred_sequences_reverse[i]

            final_peptide = pred_peptide if pred_score >= pred_score_reverse else pred_peptide_reverse
            final_score = pred_score if pred_score >= pred_score_reverse else pred_score_reverse
            charge = int(precursors[i][1].item())
            mz = precursors[i][2].item()
            pred_mass = 0.0 if len(final_peptide) == 0 else self.tokenizer.cal_seq_mz(final_peptide, charge)
            result = [
                mz,
                charge,
                pred_score,
                pred_peptide,
                pred_score_reverse,
                pred_peptide_reverse,
                pred_mass,
                final_score,
                final_peptide,
            ]
            results.append(result)
        return results

    def predict(self):
        self.predictor.model.eval()
        loop = self.test_dataloader
        if self.is_show_progress_bar:
            loop = tqdm(loop, desc="Predicting")

        # TODO: predictor 修改，使用 numpy 和 numba 转换 token ，计算质量应该更快
        end_idx = 0
        self.idx_seq = []
        self.results = []
        with torch.no_grad():
            try:
                for batch in loop:
                    outputs = self._step_prediction(batch)
                    self.results.extend(outputs)
                    self.idx_seq.extend(list(range(end_idx, end_idx + len(outputs))))
                    end_idx += len(outputs)
            except KeyboardInterrupt:
                logger.warning("Interrupted by user. Saving partial results...")
                self.save_results()
                sys.exit(1)

            self.save_results()
            logger.info("Prediction finished.")
