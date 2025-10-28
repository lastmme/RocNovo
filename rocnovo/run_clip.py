import logging
import os
from dataclasses import asdict
from typing import Optional

import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group

from .common.build import build_train_val_dataloader, build_optimizer_scheduler
from .common.ddp import ddp_setup
from .common.utils import load_model_state_dict
from .common.config import (
    Config,
    CLIPFinetuneConfig,
    AugmentationConfig,
    TrainerDataConfig,
    CLIPModelConfig,
    TrainerConfig,
    OptimizerConfig,
    CLIPPredictorConfig,
    PredictorDataConfig,
    SpectrumEncoderConfig,
    PeptideEncoderConfig,
    generate_clip_model_config,
    SchedulerConfig,
)
from .components.loss import ClipLoss
from .clip.trainer import CLIPTrainer
from .clip.predictor import CLIPPredictor
from .clip.clip import CLIP, CLIPFinetune, CLIP_Rope
from .clip.finetune import CLIPFinetuner
from .components.lr_scheduler import get_cosine_schedule_with_warmup
from .data.datasets import DeNovoStream
from .data.dataloaders import DeNovoDataLoaderModule
from .tokenizer.peptide import PTMPeptideTokenizer
from .tokenizer.spectrum import SpectrumTokenizer
from .utils import setup_logging, set_seed

logger = logging.getLogger(__name__)


def prepare_train_obj(
    spm_tokenizer: SpectrumTokenizer,
    ptm_tokenizer: PTMPeptideTokenizer,
    trainer_config: TrainerConfig,
    model_config: CLIPModelConfig,
    data_config: TrainerDataConfig,
    optitimizer_config: OptimizerConfig,
    scheduler_config: SchedulerConfig,
    world_size: int,
    choose: str = "base",
):
    train_dataloader, val_dataloader = build_train_val_dataloader(spm_tokenizer, ptm_tokenizer, data_config, world_size)

    model_config.n_vocab = ptm_tokenizer.vocab_size
    if choose == "base":
        model = CLIP(**asdict(model_config))
    elif choose == "rope":
        model = CLIP_Rope(**asdict(model_config))

    total_steps = trainer_config.max_epochs * len(train_dataloader) // world_size
    if trainer_config.warmup_epochs is not None:
        warmup_steps = int(len(train_dataloader) * trainer_config.warmup_epochs)
    else:
        warmup_steps = None
    optimizer, scheduler = build_optimizer_scheduler(
        model, optitimizer_config, scheduler_config, total_steps, warmup_steps
    )

    return train_dataloader, val_dataloader, model, optimizer, scheduler


def prepare_funetune_obj(
    spm_tokenizer: SpectrumTokenizer,
    ptm_tokenizer: PTMPeptideTokenizer,
    trainer_config: TrainerConfig,
    finetune_config: CLIPFinetuneConfig,
    clip_model_config: CLIPModelConfig,
    clip_model_checkpoint_path: str,
    data_config: TrainerDataConfig,
    optitimizer_config: OptimizerConfig,
    scheduler_config: SchedulerConfig,
    world_size: int,
):
    train_dataloader, val_dataloader = build_train_val_dataloader(spm_tokenizer, ptm_tokenizer, data_config, world_size)

    clip_model_config.n_vocab = ptm_tokenizer.vocab_size
    clip_model = CLIP(**asdict(clip_model_config))
    clip_model = load_model_state_dict(clip_model, clip_model_checkpoint_path)

    model = CLIPFinetune(clip_model, **asdict(finetune_config))

    total_steps = trainer_config.max_epochs * len(train_dataloader) // world_size
    if trainer_config.warmup_epochs is not None:
        warmup_steps = int(len(train_dataloader) * trainer_config.warmup_epochs)
    else:
        warmup_steps = None
    optimizer, scheduler = build_optimizer_scheduler(
        model, optitimizer_config, scheduler_config, total_steps, warmup_steps
    )
    return train_dataloader, val_dataloader, model, optimizer, scheduler


def _train(rank: int, world_size: int, config: Config, checkpoint_path: str):
    if rank == 0:
        log_path = os.path.join(config["trainer"]["model_save_folder"], "train.log")
        setup_logging(log_file=log_path)
        config.log_config()

    ddp_setup(rank, world_size)

    set_seed(int(config["random_seed"]))

    aug_config = AugmentationConfig(**config["aug"])
    spectrum_tokenizer = SpectrumTokenizer(**config["spectrum_tokenizer"])
    if aug_config.enabled:
        spectrum_tokenizer.set_aug_config(aug_config)

    peptide_tokenizer = PTMPeptideTokenizer(**config["peptide_tokenizer"])
    model_config = generate_clip_model_config(
        SpectrumEncoderConfig(**config["spectrum_encoder"]),
        PeptideEncoderConfig(**config["peptide_encoder"]),
    )
    
    choose = config["choose"]

    train_dataloader, val_dataloader, model, optimizer, lr_scheduler = prepare_train_obj(
        spm_tokenizer=spectrum_tokenizer,
        ptm_tokenizer=peptide_tokenizer,
        trainer_config=TrainerConfig(**config["trainer"]),
        model_config=model_config,
        data_config=TrainerDataConfig(**config["trainer_data"]),
        optitimizer_config=OptimizerConfig(**config["optimizer"]),
        scheduler_config=SchedulerConfig(**config["scheduler"]),
        world_size=world_size,
        choose=choose
    )
    loss = ClipLoss(**config["clip_loss"])

    trainer = CLIPTrainer(
        rank,
        model,
        TrainerConfig(**config["trainer"]),
        train_dataloader,
        val_dataloader,
        loss,
        optimizer,
        lr_scheduler,
        checkpoint_path,
        True,
    )
    trainer.train()
    destroy_process_group()


def train(config_path: str, checkpoint_path: str = None, device_ids: Optional[str] = None):
    if device_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids

    config = Config(config_path)
    world_size = torch.cuda.device_count()
    mp.spawn(
        _train,
        args=(world_size, config, checkpoint_path),
        nprocs=world_size,
    )


def _finetune(rank: int, world_size: int, config: Config, clip_model_checkpoint_path: str, checkpoint_path: str):
    clip_config = Config(config["clip_config_path"])

    if rank == 0:
        log_path = os.path.join(config["trainer"]["model_save_folder"], "train.log")
        setup_logging(log_file=log_path)

    ddp_setup(rank, world_size)

    set_seed(int(config["random_seed"]))

    aug_config = AugmentationConfig(**config["aug"])
    spectrum_tokenizer = SpectrumTokenizer(**config["spectrum_tokenizer"])
    if aug_config.enabled:
        spectrum_tokenizer.set_aug_config(aug_config)

    peptide_tokenizer = PTMPeptideTokenizer(**config["peptide_tokenizer"])

    clip_model_config = generate_clip_model_config(
        SpectrumEncoderConfig(**clip_config["spectrum_encoder"]),
        PeptideEncoderConfig(**clip_config["peptide_encoder"]),
    )

    train_dataloader, val_dataloader, model, optimizer, lr_scheduler = prepare_funetune_obj(
        spectrum_tokenizer,
        peptide_tokenizer,
        TrainerConfig(**config["trainer"]),
        CLIPFinetuneConfig(**config["finetune"]),
        clip_model_config,
        clip_model_checkpoint_path,
        TrainerDataConfig(**config["trainer_data"]),
        OptimizerConfig(**config["optimizer"]),
        SchedulerConfig(**config["scheduler"]),
        world_size,
    )
    loss = ClipLoss(**config["clip_loss"])

    finetuner = CLIPFinetuner(
        rank,
        model,
        TrainerConfig(**config["trainer"]),
        train_dataloader,
        val_dataloader,
        loss,
        optimizer,
        lr_scheduler,
        checkpoint_path,
        True,
    )
    finetuner.finetune()
    destroy_process_group()


def finetune(
    config_path: str, clip_model_checkpoint_path: str, checkpoint_path: str = None, device_ids: Optional[str] = None
):
    if device_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids

    config = Config(config_path)
    world_size = torch.cuda.device_count()
    mp.spawn(
        _finetune,
        args=(world_size, config, clip_model_checkpoint_path, checkpoint_path),
        nprocs=world_size,
    )


def prepare_predict_obj(
    spm_tokenizer: SpectrumTokenizer,
    ptm_tokenizer: PTMPeptideTokenizer,
    finetune_config: CLIPFinetuneConfig,
    clip_model_config: CLIPModelConfig,
    data_config: PredictorDataConfig,
    checkpoint_path: str,
):
    test_dataset = DeNovoStream(
        data_config.test_path,
        spm_tokenizer,
        ptm_tokenizer,
    )
    num_workers = min(os.cpu_count(), data_config.data_loader_workers)

    dataloader_module = DeNovoDataLoaderModule(data_config.batch_size, data_config.batch_size, num_workers)

    test_dataloader = dataloader_module.test_dataloader(test_dataset)
    clip_model_config.n_vocab = ptm_tokenizer.vocab_size
    model = CLIP(**asdict(clip_model_config))

    if finetune_config is not None:
        model = CLIPFinetune(model, **asdict(finetune_config))
        model = load_model_state_dict(model, checkpoint_path)
    else:
        model = load_model_state_dict(model, checkpoint_path)

    return test_dataloader, model


def predict(
    config_path: str, checkpoint_path: str, output_path: str, device_id: str = "0", use_finetune_model: bool = False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    log_path = output_path.replace(".csv", "_predict.log")
    setup_logging(log_file=log_path)
    logger.info(f"Config Path: {config_path}")

    config = Config(config_path)
    spectrum_tokenizer = SpectrumTokenizer(**config["spectrum_tokenizer"])
    peptide_tokenizer = PTMPeptideTokenizer(**config["peptide_tokenizer"])
    clip_config = config
    if use_finetune_model:
        clip_config = Config(config["clip_config_path"])

    clip_model_config = generate_clip_model_config(
        SpectrumEncoderConfig(**clip_config["spectrum_encoder"]),
        PeptideEncoderConfig(**clip_config["peptide_encoder"]),
    )

    finetune_config = None
    if use_finetune_model:
        finetune_config = CLIPFinetuneConfig(**config["finetune"])

    test_dataloader, model = prepare_predict_obj(
        spectrum_tokenizer,
        peptide_tokenizer,
        finetune_config,
        clip_model_config,
        PredictorDataConfig(**config["predictor_data"]),
        checkpoint_path,
    )

    predictor = CLIPPredictor(
        0,
        model,
        CLIPPredictorConfig(**config["predictor"]),
        peptide_tokenizer,
        test_dataloader,
    )
    predictor.predict(output_path)
