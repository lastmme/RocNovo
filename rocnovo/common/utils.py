import random

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from .config import Checkpoint

def set_seed(seed):
    # seed = seed * dist.get_rank()
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_model_state_dict(model: nn.Module, path: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint = Checkpoint(**checkpoint)
    model.load_state_dict(checkpoint.model_state_dict)
    return model