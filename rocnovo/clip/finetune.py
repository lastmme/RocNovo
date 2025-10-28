import logging
from datetime import datetime

from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..components.loss import ClipLoss
from ..common.config import TrainerConfig
from .trainer import CLIPTrainer
from .clip import CLIPFinetune

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

logger = logging.getLogger(__name__)

class CLIPFinetuner(CLIPTrainer):
    def __init__(
        self,
        gpu_id: int,
        model: CLIPFinetune,
        config: TrainerConfig,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        criterion: ClipLoss,
        optimizer: Optimizer,
        scheduler: LRScheduler = None,
        checkpoint_path: str = None,
        is_maximum: bool = False,
    ) -> None:
        super().__init__(
            gpu_id,
            model,
            config,
            train_dataloader,
            val_dataloader,
            criterion,
            optimizer,
            scheduler,
            checkpoint_path,
            is_maximum
        )
    
    def finetune(self):
        super().train()