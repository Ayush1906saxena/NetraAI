"""
Learning rate schedulers with warmup support.
"""

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingWarmRestarts


class WarmupCosineScheduler(_LRScheduler):
    """
    Linear warmup followed by cosine annealing decay.

    During warmup (steps 0..warmup_steps-1), LR increases linearly from
    warmup_lr to base_lr. After warmup, LR follows cosine decay from
    base_lr to min_lr over the remaining steps.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of warmup steps (iterations or epochs).
        total_steps: Total number of steps.
        warmup_lr: Starting LR during warmup (default 1e-7).
        min_lr: Minimum LR at the end of cosine decay (default 1e-7).
        last_epoch: Index of the last epoch (for resumption).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        warmup_lr: float = 1e-7,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup
            alpha = step / max(1, self.warmup_steps)
            return [
                self.warmup_lr + alpha * (base_lr - self.warmup_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr + cosine_factor * (base_lr - self.min_lr)
                for base_lr in self.base_lrs
            ]


class CosineWithWarmRestarts(_LRScheduler):
    """
    Wraps PyTorch's CosineAnnealingWarmRestarts with a linear warmup phase.

    Provides the SGDR (Stochastic Gradient Descent with Warm Restarts)
    schedule with an initial warmup ramp. Useful for fine-tuning pretrained
    models where a cold start can destabilize early training.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of warmup steps.
        T_0: Number of steps for the first cosine cycle.
        T_mult: Factor to increase cycle length after each restart.
        eta_min: Minimum LR for cosine annealing.
        warmup_lr: Starting LR during warmup.
        last_epoch: Index of the last epoch.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 1e-7,
        warmup_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        self._base_lrs_cache = [pg["lr"] for pg in optimizer.param_groups]

        # Create the inner cosine scheduler (will be stepped after warmup)
        self.cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup
            alpha = step / max(1, self.warmup_steps)
            return [
                self.warmup_lr + alpha * (base_lr - self.warmup_lr)
                for base_lr in self._base_lrs_cache
            ]
        else:
            # Delegate to cosine warm restarts
            # Manually compute the cosine schedule at the adjusted step
            adjusted_step = step - self.warmup_steps
            T_0 = self.cosine_scheduler.T_0
            T_mult = self.cosine_scheduler.T_mult
            eta_min = self.cosine_scheduler.eta_min

            if T_mult == 1:
                T_cur = adjusted_step % T_0
                T_i = T_0
            else:
                # Find which cycle we are in
                n = int(math.log((adjusted_step / T_0) * (T_mult - 1) + 1, T_mult))
                T_i = T_0 * (T_mult ** n)
                cycle_start = T_0 * (T_mult ** n - 1) // (T_mult - 1) if T_mult > 1 else T_0 * n
                T_cur = adjusted_step - cycle_start

            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * T_cur / T_i))
            return [
                eta_min + cosine_factor * (base_lr - eta_min)
                for base_lr in self._base_lrs_cache
            ]

    def step(self, epoch=None):
        """Override step to keep inner cosine scheduler in sync."""
        super().step(epoch)
