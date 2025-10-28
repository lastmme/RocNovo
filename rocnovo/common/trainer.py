import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp.grad_scaler import GradScaler
from torchinfo import summary

from .config import TrainerConfig, Checkpoint

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    def __init__(
        self,
        gpu_id: int,
        model: nn.Module,
        config: TrainerConfig,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler = None,
        checkpoint_path: str = None,
    ) -> None:
        """Initialize a ModelRunner"""
        self.run_epoch = 0
        self.run_step = 0

        self.gpu_id = gpu_id
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model

        self.criterion = criterion
        self.optimizer = optimizer
        self.is_clipgrad = False if config.grad_norm_clip is None else True
        self.scheduler = scheduler
        self.config = config

        self.tb_summarywriter = None
        if self.config.summarywriter is not None and self.is_master:
            tb_dir = Path(config.summarywriter) / TIMESTAMP
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tb_summarywriter = SummaryWriter(str(tb_dir))

        self.best_models = []
        self.max_best_models = config.save_top_k if config.save_top_k is not None else 1

        # self.scaler = GradScaler(enabled=self.config.is_grad_scale)

        if checkpoint_path is not None:
            self.load_from_checkpoint(checkpoint_path, config.load_mode)
        else:
            self.model = self.model.to(self.gpu_id)
            self.model = DDP(self.model, device_ids=[self.gpu_id])

        if self.is_master:
            summary(self.model, depth=3)

    @property
    def is_show_progress_bar(self):
        return self.is_master and self.config.show_progress_bar

    @property
    def is_master(self):
        return self.gpu_id == 0

    @property
    def is_write(self):
        return self.is_master and self.tb_summarywriter is not None

    def _save_checkpoint(self, save_path: str):
        checkpoint = Checkpoint(
            model_state_dict=self.model.module.state_dict(),
            run_step=self.run_step,
            run_epoch=self.run_epoch,
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=(self.scheduler.state_dict() if self.scheduler is not None else None),
        )
        torch.save(asdict(checkpoint), save_path)

    def load_from_checkpoint(self, path: str, load_mode: str):
        assert load_mode in [
            "TFS",
            "WARM_START",
            "RESUME",
        ], f"Invalid load_mode: {load_mode}. Load mode must be TFS, WARM_START or RESUME."

        # TFS 从头训练
        if load_mode == "TFS":
            if self.is_master:
                logger.info("Training from scratch (TFS). No checkpoint will be loaded.")
            return

        if not os.path.exists(path):
            logger.error(f"Checkpoint file not found at '{path}'. Cannot perform '{load_mode}'.")
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        if self.is_master:
            logger.info(f"Loading checkpoint from '{path}' with mode: '{load_mode}'...")

        checkpoint_data = torch.load(path, map_location="cpu")
        checkpoint = Checkpoint(**checkpoint_data)
        # WARM_START
        # 只加载模型权重，不改变优化器、调度器和训练进度。
        if load_mode == "WARM_START":
            self.model.load_state_dict(checkpoint.model_state_dict)
            self.model = self.model.to(self.gpu_id)
            self.model = DDP(self.model, device_ids=[self.gpu_id])
            if self.is_master:
                logger.info(f"WARM_START training from epoch {self.run_epoch}, step {self.run_step}")

        # RESUME
        # 加载完整的训练状态，包括模型、优化器、调度器和训练进度。
        elif load_mode == "RESUME":
            self.model.load_state_dict(checkpoint.model_state_dict)

            # 需要先将模型加载到 GPU 上，否则优化器和调度器指向的参数会在 CPU 中
            self.model = self.model.to(self.gpu_id)
            self.model = DDP(self.model, device_ids=[self.gpu_id])

            self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)

            if self.scheduler is not None and checkpoint.scheduler_state_dict is not None:
                self.scheduler.load_state_dict(checkpoint.scheduler_state_dict)

            self.run_epoch = checkpoint.run_epoch
            self.run_step = checkpoint.run_step

            if self.is_master:
                logger.info(f"Resumed training from epoch {self.run_epoch}, step {self.run_step}.")

    @abstractmethod
    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        pass

    @abstractmethod
    def _run_step(self, batch, train: bool = True):
        pass

    @abstractmethod
    def train(self) -> None:
        pass
