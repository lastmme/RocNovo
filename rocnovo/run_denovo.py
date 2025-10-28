import logging
import os
from dataclasses import asdict
from typing import Optional, Union

import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group

from .common.build import build_train_val_dataloader, build_optimizer_scheduler, build_valid_dataloader, build_bidirect_train_val_dataloader
from .common.ddp import ddp_setup
from .common.utils import load_model_state_dict

from .common.config import (
    Config,
    CLIPModelConfig,
    SpectrumEncoderConfig,
    PeptideEncoderConfig,
    AugmentationConfig,
    generate_clip_model_config,
    TransformerConfig,
    MoeConfig,
    TrainerConfig,
    TrainerDataConfig,
    OptimizerConfig,
    DecoderPredictorConfig,
    PredictorDataConfig,
    SchedulerConfig,
)
from .components.loss import DecoderLoss
from .denovo.trainer import BiDirectDecoderTrainer
from .denovo.predictor import DenovoPredictor
from .clip.clip import CLIP
from .denovo.denovo import BaseModel, BiDirectRopeClipModel
from .data.datasets import BiDirectDeNovoStream, SpectrumStream, DeNovoStream
from .data.dataloaders import BiDirectDeNovoDataLoaderModule, SpectraDataLoaderModule
from .tokenizer.peptide import PTMPeptideTokenizer, PAD
from .tokenizer.spectrum import SpectrumTokenizer
from .utils import setup_logging, set_seed

logger = logging.getLogger(__name__)

DECODER_CLASSES = {
    "BaseModel": BaseModel,
    "BiDirectRopeClipModel": BiDirectRopeClipModel,
    # "RopeClipModel": RopeClipModel,
    # "RopeModel": RopeModel,
    # "BaselineModel": BaselineModel,
}
DECODER_CONFIG_CLASSES = {
    "BaseModel": TransformerConfig,
    "BiDirectRopeClipModel": TransformerConfig,
    "RopeClipModel": TransformerConfig,
    "RopeModel": TransformerConfig,
    "BaselineModel": TransformerConfig,
}


def train(
    denovo_config_path: str,
    denovo_checkpoint_path: str = None,
    device_ids: Optional[str] = None,
):
    if device_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids

    denovo_config = Config(denovo_config_path)
    clip_config_path = denovo_config["clip_config_path"]
    clip_config = Config(clip_config_path)
    clip_checkpoint_path = denovo_config["clip_checkpoint_path"]
    world_size = torch.cuda.device_count()
    mp.spawn(
        _train,
        args=(world_size, clip_config, denovo_config, clip_checkpoint_path, denovo_checkpoint_path),
        nprocs=world_size,
    )


def prepare_train_obj(
    spm_tokenizer: SpectrumTokenizer,
    ptm_tokenizer: PTMPeptideTokenizer,
    trainer_config: TrainerConfig,
    clip_model_config: CLIPModelConfig,
    clip_checkpoint_path: str,
    choosed_decoder: str,
    decoder_model_config: Union[TransformerConfig, MoeConfig],
    data_config: TrainerDataConfig,
    optitimizer_config: OptimizerConfig,
    scheduler_config: SchedulerConfig,
    world_size: int,
):
    train_dataloader, val_dataloader = build_bidirect_train_val_dataloader(spm_tokenizer, ptm_tokenizer, data_config, world_size)
    clip_model_config.n_vocab = ptm_tokenizer.vocab_size
    clip_model = CLIP(**asdict(clip_model_config))
    if clip_checkpoint_path is not None:
        clip_model = load_model_state_dict(clip_model, clip_checkpoint_path)
    clip_model_dim = clip_model_config.embed_dim

    decoder_model_config.tokenizer = ptm_tokenizer
    decoder_class = DECODER_CLASSES[choosed_decoder]
    decoder_model = decoder_class(clip_model, clip_model_dim, **asdict(decoder_model_config))

    total_steps = trainer_config.max_epochs * len(train_dataloader) // world_size
    if trainer_config.warmup_epochs is not None:
        warmup_steps = int(len(train_dataloader) * trainer_config.warmup_epochs)
    else:
        warmup_steps = None
    optimizer, scheduler = build_optimizer_scheduler(
        decoder_model, optitimizer_config, scheduler_config, total_steps, warmup_steps
    )
    return train_dataloader, val_dataloader, decoder_model, optimizer, scheduler


def _train(
    rank: int,
    world_size: int,
    clip_config: Config,
    denovo_config: Config,
    clip_checkpoint_path: str,
    checkpoint_path: str = None,
):
    if rank == 0:
        log_path = os.path.join(denovo_config["trainer"]["model_save_folder"], "train.log")
        setup_logging(log_file=log_path)
        clip_config.log_config()
        denovo_config.log_config()

    ddp_setup(rank, world_size)

    set_seed(int(denovo_config["random_seed"]))

    # aug_config = AugmentationConfig(**clip_config["aug"])
    spectrum_tokenizer = SpectrumTokenizer(**clip_config["spectrum_tokenizer"])
    spectrum_tokenizer.disable_aug()
    peptide_tokenizer = PTMPeptideTokenizer(**clip_config["peptide_tokenizer"])

    clip_model_config = generate_clip_model_config(
        SpectrumEncoderConfig(**clip_config["spectrum_encoder"]),
        PeptideEncoderConfig(**clip_config["peptide_encoder"]),
    )

    choosed_decoder = denovo_config["DecoderModel"]["choose"]
    decoder_config_class = DECODER_CONFIG_CLASSES[choosed_decoder]
    decoder_model_config = decoder_config_class(**denovo_config["DecoderModel"][choosed_decoder])

    train_dataloader, val_dataloader, decoder_model, optimizer, lr_scheduler = prepare_train_obj(
        spm_tokenizer=spectrum_tokenizer,
        ptm_tokenizer=peptide_tokenizer,
        trainer_config=TrainerConfig(**denovo_config["trainer"]),
        clip_model_config=clip_model_config,
        clip_checkpoint_path=clip_checkpoint_path,
        choosed_decoder=choosed_decoder,
        decoder_model_config=decoder_model_config,
        data_config=TrainerDataConfig(**denovo_config["trainer_data"]),
        optitimizer_config=OptimizerConfig(**denovo_config["optimizer"]),
        scheduler_config=SchedulerConfig(**denovo_config["scheduler"]),
        world_size=world_size,
    )
    loss = DecoderLoss(peptide_tokenizer.vocab2idx[PAD], **denovo_config["decoder_loss"])
    trainer = BiDirectDecoderTrainer(
        rank,
        decoder_model,
        TrainerConfig(**denovo_config["trainer"]),
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


def predict(denovo_config_path: str, denovo_checkpoint_path: str, output_path: str, device_id: str = "0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    log_path = output_path.replace(".csv", "_predict.log")
    setup_logging(log_file=log_path)
    logger.info(f"Config Path: {denovo_config_path}")

    denovo_config = Config(denovo_config_path)
    clip_config_path = denovo_config["clip_config_path"]
    clip_config = Config(clip_config_path)
    clip_checkpoint_path = denovo_config["clip_checkpoint_path"]
    clip_config.log_config()
    denovo_config.log_config()

    spectrum_tokenizer = SpectrumTokenizer(**clip_config["spectrum_tokenizer"])
    peptide_tokenizer = PTMPeptideTokenizer(**clip_config["peptide_tokenizer"])

    clip_model_config = generate_clip_model_config(
        SpectrumEncoderConfig(**clip_config["spectrum_encoder"]),
        PeptideEncoderConfig(**clip_config["peptide_encoder"]),
    )

    choosed_decoder = denovo_config["DecoderModel"]["choose"]
    decoder_config_class = DECODER_CONFIG_CLASSES[choosed_decoder]
    decoder_model_config = decoder_config_class(**denovo_config["DecoderModel"][choosed_decoder])

    test_dataloader, decoder_model = prepare_predict_obj(
        spm_tokenizer=spectrum_tokenizer,
        ptm_tokenizer=peptide_tokenizer,
        clip_model_config=clip_model_config,
        clip_checkpoint_path=clip_checkpoint_path,
        choosed_decoder=choosed_decoder,
        decoder_model_config=decoder_model_config,
        data_config=PredictorDataConfig(**denovo_config["predictor_data"]),
    )
    predictor = DenovoPredictor(
        0,
        decoder_model,
        DecoderPredictorConfig(**denovo_config["predictor"]),
        peptide_tokenizer,
        test_dataloader,
        denovo_checkpoint_path,
        output_path,
    )
    predictor.predict()


def prepare_predict_obj(
    spm_tokenizer: SpectrumTokenizer,
    ptm_tokenizer: PTMPeptideTokenizer,
    clip_model_config: CLIPModelConfig,
    clip_checkpoint_path: str,
    choosed_decoder: str,
    decoder_model_config: Union[TransformerConfig, MoeConfig],
    data_config: PredictorDataConfig,
):
    spm_tokenizer.disable_aug()
    test_dataset = SpectrumStream(
        data_config.test_path,
        spm_tokenizer,
    )
    num_workers = min(os.cpu_count(), data_config.data_loader_workers)

    dataloader_module = SpectraDataLoaderModule(data_config.batch_size, data_config.batch_size, num_workers)

    test_dataloader = dataloader_module.test_dataloader(test_dataset)
    clip_model_config.n_vocab = ptm_tokenizer.vocab_size
    clip_model = CLIP(**asdict(clip_model_config))
    if clip_checkpoint_path is not None:
        clip_model = load_model_state_dict(clip_model, clip_checkpoint_path)
    clip_model_dim = clip_model_config.embed_dim

    decoder_model_config.tokenizer = ptm_tokenizer
    decoder_class = DECODER_CLASSES[choosed_decoder]
    decoder_model = decoder_class(clip_model, clip_model_dim, **asdict(decoder_model_config))
    return test_dataloader, decoder_model
