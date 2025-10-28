from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.05,
    ):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU()
        self.temperature = temperature

    def _anchor_dot_contrast(self, m1: torch.Tensor, m2: torch.Tensor):
        batch_size = m1.shape[0]
        m1 = F.normalize(m1, dim=-1)
        m2 = F.normalize(m2, dim=-1)
        logits = torch.div(self.leaky_relu(torch.matmul(m1, m2.T)), self.temperature)
        # 减去最大值，保证后续 softmax 数值稳定
        # 因为后续会进行 exp 的除法，抵消了最大值的影响, 不影响最终的结果
        logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits)
        # 计算最大似然概率, 和 crossEntropy 一致
        mask = torch.eye(batch_size, dtype=torch.float32, device=m1.device)
        # log(a/b) = log(a) - log(b)
        # log[exp(logits) / sum(exp(logits))] = logits - log(sum(exp(logits)))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos

        return loss.mean()

    def forward(self, logits1: torch.Tensor, logits2: torch.Tensor):
        loss1 = self._anchor_dot_contrast(logits1, logits2)
        loss2 = self._anchor_dot_contrast(logits2, logits1)
        return (loss1 + loss2) / 2


class TopkAccuracy(nn.Module):
    def __init__(self, topk: int = 1):
        super().__init__()
        self.topk = topk

    def forward(self, logits1: torch.Tensor, logits2: torch.Tensor):
        batch_size = logits1.shape[0]
        logits1 = F.normalize(logits1, dim=-1)
        logits2 = F.normalize(logits2, dim=-1)
        score = torch.matmul(logits1, logits2.T)
        _, indices = torch.topk(score, k=self.topk, dim=-1)
        labels = torch.arange(batch_size, dtype=torch.long, device=logits1.device)
        acc = (indices == labels.view(-1, 1)).any(dim=-1).float().mean()
        return acc


class DecoderLoss(nn.Module):
    def __init__(self, pad_index: int, label_smoothing: float = 0.0):
        super().__init__()
        self.pad_index = pad_index
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        V = logits.shape[-1]
        logits_flat = logits.reshape(-1, V)  # (B*L, V)
        targets_flat = targets.reshape(-1)  # (B*L,)

        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.pad_index,
            reduction="mean",
            label_smoothing=self.label_smoothing,
        )
        return loss
