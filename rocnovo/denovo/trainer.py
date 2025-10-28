import random
import logging
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from collections import deque

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch.distributed as dist

from ..common.config import TrainerConfig
from ..common.trainer import BaseTrainer
from ..components.loss import DecoderLoss

logger = logging.getLogger(__name__)
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())


class BiDirectDecoderTrainer(BaseTrainer):
    def __init__(
        self,
        gpu_id: int,
        model: torch.nn.Module,
        config: TrainerConfig,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        criterion: DecoderLoss,
        optimizer: Optimizer,
        scheduler: LRScheduler = None,
        checkpoint_path: str = None,
        is_maximum: bool = True,
    ):
        """Initialize a ModelRunner"""
        super().__init__(
            gpu_id, model, config, train_dataloader, val_dataloader, criterion, optimizer, scheduler, checkpoint_path
        )
        self.is_maximum = is_maximum
        self.world_size = dist.get_world_size()

        self.recent_loss_10000 = deque(maxlen=10000)

    def top1_full_peptide_match_accuracy(self, logits: torch.Tensor, tgt: torch.Tensor, tgt_padding_mask: torch.Tensor):
        pred = torch.argmax(logits, dim=-1)
        matches = ((pred == tgt) | tgt_padding_mask).all(dim=-1)  # 忽略 padding 的位置，默认padding为true
        return torch.mean(matches.float())

    def top1_aa_match_accuracy(logits: torch.Tensor, tgt: torch.Tensor, tgt_padding_mask: torch.Tensor):
        pred = torch.argmax(logits, dim=-1)
        valid_mask = ~tgt_padding_mask
        correct = (pred == tgt) & valid_mask
        accuracy = correct.sum() / valid_mask.sum()
        return accuracy

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        # eval or train
        self.model.train(train)
        dataloader.sampler.set_epoch(epoch)

        loop = dataloader
        if self.is_show_progress_bar:
            loop = tqdm(dataloader)

        mode = "Train" if train else "Validation"

        total_loss = 0
        total_accuracy = 0
        total_accuracy_reverse = 0
        size = 0

        for batch in loop:
            batch_size = batch[0].shape[0]
            size += batch_size

            step_loss, step_accuracy, step_accuracy_reverse = self._run_step(batch, train)
            total_loss += step_loss * batch_size
            total_accuracy += step_accuracy * batch_size
            total_accuracy_reverse += step_accuracy_reverse * batch_size

            if self.is_show_progress_bar:
                loop.set_description(f"{mode} Epoch[{epoch} / {self.config.max_epochs}]")
                loop.set_postfix(
                    loss=f"{step_loss:.4f}",
                    accuracy=f"{step_accuracy:.4f}",
                    accuracy_reverse=f"{step_accuracy_reverse:.4f}",
                )

            if self.is_write and train:
                self.tb_summarywriter.add_scalar(f"{mode}/step_loss", step_loss, global_step=self.run_step)
                self.tb_summarywriter.add_scalar(f"{mode}/step_accuracy", step_accuracy, global_step=self.run_step)
                self.tb_summarywriter.add_scalar(
                    f"{mode}/step_accuracy_reverse", step_accuracy_reverse, global_step=self.run_step
                )

            if train and self.run_step % self.config.validation_steps == 0:
                val_epoch_loss, val_epoch_accuracy, val_epoch_accuracy_reverse = self._run_epoch(
                    epoch, self.val_dataloader, False
                )
                if self.is_write:
                    self.tb_summarywriter.add_scalar(
                        f"Validation/epoch_loss", val_epoch_loss, global_step=self.run_step
                    )
                    self.tb_summarywriter.add_scalar(
                        f"Validation/epoch_accuracy", val_epoch_accuracy, global_step=self.run_step
                    )
                    self.tb_summarywriter.add_scalar(
                        f"Validation/epoch_accuracy_reverse", val_epoch_accuracy_reverse, global_step=self.run_step
                    )
                # 保存最佳模型
                if self.is_master:
                    checkpoint_path = os.path.join(
                        self.config.model_save_folder,
                        f"checkpoint_step_{self.run_step}.ckpt",
                    )
                    self._save_checkpoint(checkpoint_path)
                    self.best_models.append((self.run_step, val_epoch_accuracy, checkpoint_path))
                    self.best_models.sort(key=lambda x: x[1], reverse=self.is_maximum)
                    if len(self.best_models) > self.max_best_models:
                        for _, _, path in self.best_models[self.max_best_models :]:
                            if os.path.exists(path):
                                os.remove(path)
                        self.best_models = self.best_models[: self.max_best_models]

                if self.is_master:
                    logger.info(
                        f"Epoch {self.run_epoch+1}, Step {self.run_step}: valid_loss = {val_epoch_loss:.4f}, valid_accuracy = {val_epoch_accuracy:.4f}, valid_accuracy_reverse = {val_epoch_accuracy_reverse:.4f}"
                    )

        if not train:
            self.model.train()

        loss_tensor = torch.tensor(total_loss, device=self.gpu_id)
        acc_tensor = torch.tensor(total_accuracy, device=self.gpu_id)
        acc_reverse_tensor = torch.tensor(total_accuracy_reverse, device=self.gpu_id)
        size_tensor = torch.tensor(size, device=self.gpu_id)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(size_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_reverse_tensor, op=dist.ReduceOp.SUM)
        return (
            (loss_tensor / size_tensor).item(),
            (acc_tensor / size_tensor).item(),
            (acc_reverse_tensor / size_tensor).item(),
        )

    def _run_step(self, batch, train: bool = True):
        batch = [item.to(self.gpu_id) for item in batch]

        spectra, precursors, spectra_mask, tokens, tokens_mask, tokens_reverse, tokens_reverse_mask = batch
        seq_len = tokens.shape[1]

        with torch.set_grad_enabled(train):
            logits, logits_reverse = self.model(
                spectra=spectra,
                precursors=precursors,
                tokens=tokens[:, :-1],
                tokens_reverse=tokens_reverse[:, :-1],
                src_padding_mask=spectra_mask,
                tgt_padding_mask=tokens_mask[:, :-1],
                tgt_reverse_padding_mask=tokens_reverse_mask[:, :-1],
            )

            loss: torch.Tensor = 0.5 * self.criterion(logits, tokens[:, 1:]) + 0.5 * self.criterion(
                logits_reverse, tokens_reverse[:, 1:]
            )
            accuracy = self.top1_full_peptide_match_accuracy(logits, tokens[:, 1:], tokens_mask[:, 1:])
            accuracy_reverse = self.top1_full_peptide_match_accuracy(
                logits_reverse, tokens_reverse[:, 1:], tokens_reverse_mask[:, 1:]
            )

        if train:
            self.optimizer.zero_grad()
            loss.backward()

            if self.is_clipgrad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_norm_clip)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.run_step += 1

        return loss.item(), accuracy.item(), accuracy_reverse.item()

    def train(self) -> None:
        if self.is_master:
            logger.info(f"TB Path: {self.tb_summarywriter.log_dir}")
            logger.info(f"Begin Step : {self.run_step}  Begin Epoch: {self.run_epoch + 1}")
        for epoch in range(self.run_epoch, self.config.max_epochs):
            train_epoch_loss, train_epoch_accuracy, train_epoch_accuracy_reverse = self._run_epoch(
                epoch + 1, self.train_dataloader
            )

            if self.is_write:
                self.tb_summarywriter.add_scalar("Train/epoch_loss", train_epoch_loss, self.run_epoch + 1)
                self.tb_summarywriter.add_scalar("Train/epoch_accuracy", train_epoch_accuracy, self.run_epoch + 1)
                self.tb_summarywriter.add_scalar(
                    "Train/epoch_accuracy_reverse", train_epoch_accuracy_reverse, self.run_epoch + 1
                )

            if self.is_master:
                logger.info(
                    f"Epoch {self.run_epoch+1}, Step {self.run_step}: train_loss = {train_epoch_loss:.4f}, train_accuracy = {train_epoch_accuracy:.4f}, train_accuracy_reverse = {train_epoch_accuracy_reverse:.4f}"
                )

            self.run_epoch += 1
