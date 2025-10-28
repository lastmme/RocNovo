import math
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
            optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
            num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
            num_training_steps (:obj:`int`):
            The total number of training steps.
            num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
            last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
            :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_restart_cosine_decay_scheduler_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    lr_decay_factor: float = 5.0,
    last_epoch: int = -1,
):
    """
    Creates a scheduler that combines a linear warmup, cosine restarts, and a
    decaying peak learning rate after each restart.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        num_cycles (int, optional, defaults to 1):
            The number of cosine cycles (restarts) to perform.
        lr_decay_factor (float, optional, defaults to 5.0):
            The factor by which the peak learning rate is divided after each cycle.
        last_epoch (int, optional, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        torch.optim.lr_scheduler.LambdaLR: A scheduler with the specified behavior.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress_after_warmup = float(current_step - num_warmup_steps)
        steps_per_cycle = float(max(1, num_training_steps - num_warmup_steps)) / float(max(1, num_cycles))

        current_cycle = math.floor(progress_after_warmup / steps_per_cycle)
        peak_lr_scale = 1.0 / (lr_decay_factor**current_cycle)
        progress_in_cycle = (progress_after_warmup % steps_per_cycle) / steps_per_cycle
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress_in_cycle))

        return max(0.0, peak_lr_scale * cosine_scale)
    return LambdaLR(optimizer, lr_lambda, last_epoch)
