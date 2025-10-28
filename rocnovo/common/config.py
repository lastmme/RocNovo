from typing import Dict, Any, Union
from collections import OrderedDict
from dataclasses import dataclass
from pprint import pformat

import yaml
import torch
import logging

from ..tokenizer.peptide import SPECIAL_TOKENS, CANONICAL, PTMPeptideTokenizer

logger = logging.getLogger(__name__)

"""
 Model
"""


@dataclass
class TransformerConfig:
    model_dim: int
    n_layers: int
    n_heads: int
    dim_feedforward: int
    dropout: float
    tokenizer: PTMPeptideTokenizer = PTMPeptideTokenizer("massivekb", reverse=True)


@dataclass
class MoeConfig:
    model_dim: int
    n_layers: int
    n_heads: int
    n_experts: int
    n_activated_experts: int
    load_balancing_coeff: float
    dim_feedforward: int
    dropout: float
    tokenizer: PTMPeptideTokenizer = PTMPeptideTokenizer("massivekb", reverse=True)


@dataclass
class ConformerConfig:
    model_dim: int
    n_layers: int
    n_heads: int
    dim_feedforward: int
    conv_kernel_size: int
    dropout: float
    tokenizer: PTMPeptideTokenizer = PTMPeptideTokenizer("massivekb", reverse=True)


@dataclass
class DecoderLossConfig:
    label_smoothing: float = 0.0


@dataclass
class DecoderPredictorConfig:
    show_progress_bar: bool
    beam_search: Dict[str, Any]


@dataclass
class BeamsearchConfig:
    beam_size: int
    max_len: int
    mass_tolerance: int
    max_isotope: int
    min_peptide_len: int
    n_vocab: int = len(SPECIAL_TOKENS) + len(CANONICAL)


"""Tokenizer"""


@dataclass
class SpectrumTokenizerConfig:
    n_top_peaks: int
    min_mz: float
    max_mz: float
    min_intensity: float
    remove_precursor_tol: float


@dataclass
class PeptideTokenizerConfig:
    reverse: bool = False
    residues: Union[Dict, str] = "massivekb"


"""Data Augmentation"""


@dataclass
class AugmentationConfig:
    enabled: bool  # 是否启用数据增强
    prob: float = 0.5  # 数据增强的概率 (random >= p 则不启用)
    removal_rate: float = 0.2  # 最多遮掩峰的比例
    removal_intensity_threshold: float = 0.3  # 仅遮掩小于该阈值的峰
    perturbation_rate: float = 0.15  # 强度偏移的比例


"""Encoders"""


@dataclass
class SpectrumEncoderConfig:
    embed_dim: int
    n_heads: int
    n_layers: int
    dropout: float
    feedforward: int


@dataclass
class PeptideEncoderConfig:
    embed_dim: int
    n_heads: int
    n_layers: int
    dropout: float
    feedforward: int
    n_vocab: int = len(CANONICAL) + len(SPECIAL_TOKENS)


"""CLIP Model"""


@dataclass
class CLIPModelConfig:
    embed_dim: int
    Spectrum_heads: int
    Spectrum_layers: int
    Spectrum_dropout: float
    Spectrum_feedforward: int
    n_vocab: int
    Peptide_heads: int
    Peptide_layers: int
    Peptide_dropout: float
    Peptide_feedforward: int


@dataclass
class CLIPFinetuneConfig:
    dim_input: 1024
    dim_hidden: 4096
    dim_output: 2048


@dataclass
class ClipLossConfig:
    temperature: float


@dataclass
class CLIPPredictorConfig:
    show_progress_bar: bool


"""
Trainer 配置部分
"""


@dataclass
class TrainerDataConfig:
    train_path: str
    valid_path: str
    batch_size: int
    data_loader_workers: int


@dataclass
class TrainerConfig:
    max_epochs: int
    model_save_folder: str
    validation_steps: int
    show_progress_bar: bool
    n_log: int
    warmup_epochs: int = None  # 如果 yaml 配置文件中不含 warmup_epochs 参数，则设为默认不进行预热
    grad_norm_clip: float = None  # 如果 yaml 配置文件中不含 grad_norm_clip 参数，则设为默认不进行梯度裁剪
    save_top_k: int = None
    summarywriter: str = None
    is_grad_scale: bool = True  # 混合精度训练
    load_mode: str = "TFS" # TFS, WARM_START, RESUME 分别代表从头训练、只加载权重、加载权重和优化器和调度器状态


@dataclass
class OptimizerConfig:
    learning_rate: float
    weight_decay: float


@dataclass
class SchedulerConfig:
    num_cycles: int = 1
    lr_decay_factor: float = 1.0


"""
Predictor 配置部分
"""


@dataclass
class PredictorDataConfig:
    test_path: str
    batch_size: int
    data_loader_workers: int


"""
需要保存的训练状态
"""


@dataclass
class Checkpoint:
    model_state_dict: "OrderedDict[str, torch.Tensor]"
    run_step: int
    run_epoch: int
    optimizer_state_dict: Dict[str, Any]
    scheduler_state_dict: Dict[str, Any] = None


def generate_clip_model_config(
    spectrum_encoder_config: SpectrumEncoderConfig,
    peptide_encoder_config: PeptideEncoderConfig,
):
    if spectrum_encoder_config.embed_dim != peptide_encoder_config.embed_dim:
        raise ValueError("Spectrum and Peptide encoder embed dim must be the same")

    return CLIPModelConfig(
        embed_dim=spectrum_encoder_config.embed_dim,
        Spectrum_heads=spectrum_encoder_config.n_heads,
        Spectrum_layers=spectrum_encoder_config.n_layers,
        Spectrum_dropout=spectrum_encoder_config.dropout,
        Spectrum_feedforward=spectrum_encoder_config.feedforward,
        n_vocab=peptide_encoder_config.n_vocab,
        Peptide_heads=peptide_encoder_config.n_heads,
        Peptide_layers=peptide_encoder_config.n_layers,
        Peptide_dropout=peptide_encoder_config.dropout,
        Peptide_feedforward=peptide_encoder_config.feedforward,
    )


class Config:
    def __init__(self, path: str):
        with open(path, "r") as f:
            self.config = yaml.safe_load(f)

    def log_config(self):
        logger.info("\n" + pformat(self.config, width=500))

    def __getitem__(self, name: str) -> Any:
        return self.config[name]

    def __repr__(self):
        return repr(self.config)
