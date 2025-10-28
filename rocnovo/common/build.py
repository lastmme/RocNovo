import os
from copy import copy

import torch

from ..common.config import TrainerDataConfig, OptimizerConfig, PredictorDataConfig, SchedulerConfig
from ..components.lr_scheduler import (
    get_cosine_schedule_with_warmup,
    get_restart_cosine_decay_scheduler_with_warmup,
)
from ..data.datasets import BiDirectDeNovoStream, DeNovoStream
from ..data.dataloaders import BiDirectDeNovoDataLoaderModule, DeNovoDataLoaderModule
from ..tokenizer.peptide import PTMPeptideTokenizer
from ..tokenizer.spectrum import SpectrumTokenizer


def build_train_val_dataloader(
    spm_tokenizer: SpectrumTokenizer,
    ptm_tokenizer: PTMPeptideTokenizer,
    data_config: TrainerDataConfig,
    world_size: int,
):
    # 验证集不采用数据增强
    spm_tokenizer_val = copy(spm_tokenizer)
    spm_tokenizer_val.disable_aug()

    train_dataset = DeNovoStream(
        data_config.train_path,
        spm_tokenizer,
        ptm_tokenizer,
    )

    val_dataset = DeNovoStream(
        data_config.valid_path,
        spm_tokenizer_val,
        ptm_tokenizer,
    )

    num_workers = min(os.cpu_count() // world_size, data_config.data_loader_workers)
    dataloader_module = DeNovoDataLoaderModule(
        data_config.batch_size,
        data_config.batch_size,
        num_workers,
    )
    train_dataloader = dataloader_module.train_dataloader(train_dataset)
    val_dataloader = dataloader_module.val_dataloader(val_dataset)

    return train_dataloader, val_dataloader


def build_bidirect_train_val_dataloader(
    spm_tokenizer: SpectrumTokenizer,
    ptm_tokenizer: PTMPeptideTokenizer,
    data_config: TrainerDataConfig,
    world_size: int,
):
    # 验证集不采用数据增强，只对 clip 训练有用
    spm_tokenizer_val = copy(spm_tokenizer)
    spm_tokenizer_val.disable_aug()

    train_dataset = BiDirectDeNovoStream(
        data_config.train_path,
        spm_tokenizer,
        ptm_tokenizer,
    )

    val_dataset = BiDirectDeNovoStream(
        data_config.valid_path,
        spm_tokenizer_val,
        ptm_tokenizer,
    )

    num_workers = min(os.cpu_count() // world_size, data_config.data_loader_workers)
    dataloader_module = BiDirectDeNovoDataLoaderModule(
        data_config.batch_size,
        data_config.batch_size,
        num_workers,
    )
    train_dataloader = dataloader_module.train_dataloader(train_dataset)
    val_dataloader = dataloader_module.val_dataloader(val_dataset)

    return train_dataloader, val_dataloader


def build_test_dataloader(
    spm_tokenizer: SpectrumTokenizer,
    ptm_tokenizer: PTMPeptideTokenizer,
    data_config: PredictorDataConfig,
):
    test_dataset = BiDirectDeNovoStream(
        data_config.test_path,
        spm_tokenizer,
        ptm_tokenizer,
    )
    num_workers = min(os.cpu_count(), data_config.data_loader_workers)

    dataloader_module = BiDirectDeNovoDataLoaderModule(data_config.batch_size, data_config.batch_size, num_workers)

    test_dataloader = dataloader_module.test_dataloader(test_dataset)
    return test_dataloader


def build_valid_dataloader(
    spm_tokenizer: SpectrumTokenizer,
    ptm_tokenizer: PTMPeptideTokenizer,
    data_config: TrainerDataConfig,
    valid_path: str = None,
):
    valid_dataset = BiDirectDeNovoStream(
        valid_path if valid_path else data_config.valid_path,
        spm_tokenizer,
        ptm_tokenizer,
    )
    num_workers = min(os.cpu_count(), data_config.data_loader_workers)

    dataloader_module = BiDirectDeNovoDataLoaderModule(data_config.batch_size, data_config.batch_size, num_workers)

    valid_dataloader = dataloader_module.val_dataloader(valid_dataset)
    return valid_dataloader


def build_optimizer_scheduler(
    model: torch.nn.Module,
    optitimizer_config: OptimizerConfig,
    scheduler_config: SchedulerConfig,
    total_steps: int,
    warmup_steps: int = None,
):

    if warmup_steps is None:
        warmup_steps = int(total_steps * 0.05)  # 默认使用5%的warmup步数

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optitimizer_config.learning_rate,
        weight_decay=optitimizer_config.weight_decay,
    )

    lr_scheduler = get_restart_cosine_decay_scheduler_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=scheduler_config.num_cycles,
        lr_decay_factor=scheduler_config.lr_decay_factor,
    )
    return optimizer, lr_scheduler
