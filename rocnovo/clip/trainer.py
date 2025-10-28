import os
import logging

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch.distributed as dist
from torch.distributed.nn import all_gather
from torch.amp.autocast_mode import autocast

from .clip import CLIP
from ..common.config import TrainerConfig
from ..components.loss import ClipLoss
from ..common.trainer import BaseTrainer

logger = logging.getLogger(__name__)


class CLIPTrainer(BaseTrainer):
    def __init__(
        self,
        gpu_id: int,  # 类似于 device
        model: CLIP,
        config: TrainerConfig,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        criterion: ClipLoss,
        optimizer: Optimizer,
        scheduler: LRScheduler = None,
        checkpoint_path: str = None,
        is_maximum: bool = False,
    ) -> None:
        """Initialize a ModelRunner"""
        super().__init__(
            gpu_id, model, config, train_dataloader, val_dataloader, criterion, optimizer, scheduler, checkpoint_path
        )
        self.is_maximum = is_maximum
        self.world_size = dist.get_world_size()

    def top1_accuracy(self, logits_a: torch.Tensor, logits_b: torch.Tensor):
        logits_a_norm = F.normalize(logits_a, dim=-1)
        logits_b_norm = F.normalize(logits_b, dim=-1)
        cosine_score = torch.matmul(logits_a_norm, logits_b_norm.T)
        top1_indices = torch.argmax(cosine_score, dim=-1)
        labels = torch.arange(logits_a.size(0), device=self.gpu_id)
        accuracy = torch.mean((top1_indices == labels).float())
        return accuracy

    def step_top1_accuracy(self, logits_a: torch.Tensor, logits_b: torch.Tensor):
        forward_accuracy_top1 = self.top1_accuracy(logits_a, logits_b)
        backward_accuracy_top1 = self.top1_accuracy(logits_b, logits_a)
        accuracy = (forward_accuracy_top1 + backward_accuracy_top1) / 2
        return accuracy

    def gather_logits(self, logit_tensors: torch.Tensor):
        logit_tensors = logit_tensors.contiguous()
        gathered_logits = [torch.zeros_like(logit_tensors) for _ in range(self.world_size)]
        dist.all_gather(gathered_logits, logit_tensors)
        return gathered_logits

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
        size = 0

        for batch in loop:
            batch_size = batch[0].shape[0]
            size += batch_size

            step_loss, step_accuracy = self._run_step(batch, train)
            total_loss += step_loss * batch_size

            total_accuracy += step_accuracy * batch_size

            if self.is_show_progress_bar:
                loop.set_description(f"{mode} Epoch[{epoch} / {self.config.max_epochs}]")
                loop.set_postfix(loss=f"{step_loss:.4f}", accuracy=f"{step_accuracy:.4f}")

            if self.is_write and train:
                self.tb_summarywriter.add_scalar(f"{mode}/step_loss", step_loss, global_step=self.run_step)
                self.tb_summarywriter.add_scalar(f"{mode}/step_accuracy", step_accuracy, global_step=self.run_step)

            if train and self.run_step % self.config.validation_steps == 0:
                val_epoch_loss, val_epoch_accuracy = self._run_epoch(epoch, self.val_dataloader, False)
                if self.is_write:
                    self.tb_summarywriter.add_scalar(
                        f"Validation/epoch_loss", val_epoch_loss, global_step=self.run_step
                    )
                    self.tb_summarywriter.add_scalar(
                        f"Validation/epoch_accuracy", val_epoch_accuracy, global_step=self.run_step
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
                        f"Epoch {self.run_epoch+1}, Step {self.run_step}: valid_loss = {val_epoch_loss:.4f}, valid_accuracy = {val_epoch_accuracy:.4f}"
                    )

        if not train:
            self.model.train()

        loss_tensor = torch.tensor(total_loss, device=self.gpu_id)
        acc_tensor = torch.tensor(total_accuracy, device=self.gpu_id)
        size_tensor = torch.tensor(size, device=self.gpu_id)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(size_tensor, op=dist.ReduceOp.SUM)
        return (loss_tensor / size_tensor).item(), (acc_tensor / size_tensor).item()

    def _run_step(self, batch, train: bool = True):
        batch = [item.to(self.gpu_id) for item in batch]
        spectra, precursors, spectra_mask, tokens, tokens_mask = batch
        with torch.set_grad_enabled(train):
            logits_per_spec, logits_per_tgt = self.model(
                spectra=spectra,
                precursors=precursors,
                tokens=tokens,
                spectra_mask=spectra_mask,
                tokens_mask=tokens_mask,
            )

            global_logits_spec = torch.cat(all_gather(logits_per_spec))
            global_logits_tgt = torch.cat(all_gather(logits_per_tgt))

            loss = self.criterion(global_logits_spec, global_logits_tgt)
            accuracy = self.step_top1_accuracy(global_logits_spec, global_logits_tgt)

        # --- 4. 反向传播与优化 ---
        if train:
            self.optimizer.zero_grad()
            loss.backward()

            if self.is_clipgrad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_norm_clip)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.run_step += 1

        # 为了日志和准确率计算，返回 loss 和全局 logits
        return loss.item(), accuracy.item()

    def train(self) -> None:
        if self.is_master:
            logger.info(f"TB Path: {self.tb_summarywriter.log_dir}")
            logger.info(f"Begin Step : {self.run_step}  Begin Epoch: {self.run_epoch + 1}")
        for epoch in range(self.run_epoch, self.config.max_epochs):
            train_epoch_loss, train_epoch_accuracy = self._run_epoch(epoch + 1, self.train_dataloader)

            if self.is_write:
                self.tb_summarywriter.add_scalar("Train/epoch_loss", train_epoch_loss, self.run_epoch + 1)
                self.tb_summarywriter.add_scalar("Train/epoch_accuracy", train_epoch_accuracy, self.run_epoch + 1)

            if self.is_master:
                logger.info(
                    f"Epoch {self.run_epoch+1}, Step {self.run_step}: train_loss = {train_epoch_loss:.4f}, train_accuracy = {train_epoch_accuracy:.4f}"
                )

            self.run_epoch += 1
