"""
Training callbacks for early stopping, model checkpointing, and LR logging.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Stops training when a monitored metric has stopped improving.

    Args:
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'min' for loss-like metrics, 'max' for accuracy-like metrics.
        verbose: Whether to log messages on state changes.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
        verbose: bool = True,
    ):
        assert mode in ("min", "max"), "mode must be 'min' or 'max'"
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False
        self.best_epoch: int = 0

    def _is_improvement(self, current: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return current < self.best_score - self.min_delta
        return current > self.best_score + self.min_delta

    def __call__(self, epoch: int, metric_value: float) -> bool:
        """
        Check whether to stop training.

        Args:
            epoch: Current epoch index.
            metric_value: Value of the monitored metric.

        Returns:
            True if training should stop, False otherwise.
        """
        if self._is_improvement(metric_value):
            self.best_score = metric_value
            self.counter = 0
            self.best_epoch = epoch
            if self.verbose:
                logger.info(
                    f"EarlyStopping: metric improved to {metric_value:.6f} at epoch {epoch}"
                )
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping: no improvement for {self.counter}/{self.patience} epochs "
                    f"(best={self.best_score:.6f} at epoch {self.best_epoch})"
                )
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    logger.info(
                        f"EarlyStopping: triggered after {self.patience} epochs without improvement"
                    )

        return self.should_stop

    def reset(self):
        """Reset the callback state."""
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0


class ModelCheckpoint:
    """
    Save model checkpoints when monitored metric improves.

    Saves a full checkpoint dict containing model state, optimizer state,
    scheduler state, epoch, and metrics. Also supports saving top-K checkpoints.

    Args:
        save_dir: Directory to save checkpoints.
        monitor: Name of the metric to monitor (for filename and logging).
        mode: 'min' or 'max'.
        save_top_k: Number of best checkpoints to keep. -1 keeps all.
        save_last: Whether to always save the last epoch checkpoint.
        verbose: Whether to log messages.
    """

    def __init__(
        self,
        save_dir: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 3,
        save_last: bool = True,
        verbose: bool = True,
    ):
        assert mode in ("min", "max"), "mode must be 'min' or 'max'"
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.verbose = verbose

        # Track saved checkpoints: list of (metric_value, filepath)
        self.saved_checkpoints: list = []
        self.best_score: Optional[float] = None

    def _is_improvement(self, current: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return current < self.best_score
        return current > self.best_score

    def __call__(
        self,
        epoch: int,
        metric_value: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        extra_info: Optional[dict] = None,
    ) -> Optional[Path]:
        """
        Potentially save a checkpoint.

        Args:
            epoch: Current epoch.
            metric_value: Value of the monitored metric.
            model: The model to save.
            optimizer: The optimizer to save.
            scheduler: Optional LR scheduler to save.
            extra_info: Optional dict of additional info (metrics, config, etc.).

        Returns:
            Path to saved checkpoint if saved, else None.
        """
        saved_path = None

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric_name": self.monitor,
            "metric_value": metric_value,
        }
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if extra_info is not None:
            checkpoint["extra_info"] = extra_info

        # Save last checkpoint
        if self.save_last:
            last_path = self.save_dir / "last.pt"
            torch.save(checkpoint, last_path)

        # Check if this is a top-K checkpoint
        is_better = self._is_improvement(metric_value)
        if is_better:
            self.best_score = metric_value

        # Determine if we should save
        should_save = False
        if self.save_top_k == -1:
            should_save = True
        elif self.save_top_k > 0:
            if len(self.saved_checkpoints) < self.save_top_k:
                should_save = True
            else:
                # Check if current is better than worst saved
                worst_idx = self._get_worst_idx()
                worst_value = self.saved_checkpoints[worst_idx][0]
                if self.mode == "min" and metric_value < worst_value:
                    should_save = True
                elif self.mode == "max" and metric_value > worst_value:
                    should_save = True

        if should_save:
            filename = f"epoch={epoch:03d}_{self.monitor}={metric_value:.4f}.pt"
            filepath = self.save_dir / filename
            torch.save(checkpoint, filepath)
            saved_path = filepath

            self.saved_checkpoints.append((metric_value, filepath))

            if self.verbose:
                logger.info(f"ModelCheckpoint: saved {filepath.name} ({self.monitor}={metric_value:.4f})")

            # Remove worst if we exceed top_k
            if self.save_top_k > 0 and len(self.saved_checkpoints) > self.save_top_k:
                worst_idx = self._get_worst_idx()
                worst_value, worst_path = self.saved_checkpoints.pop(worst_idx)
                if worst_path.exists():
                    worst_path.unlink()
                    if self.verbose:
                        logger.info(f"ModelCheckpoint: removed old checkpoint {worst_path.name}")

        # Save best model symlink / copy
        if is_better:
            best_path = self.save_dir / "best.pt"
            torch.save(checkpoint, best_path)
            if self.verbose:
                logger.info(f"ModelCheckpoint: new best {self.monitor}={metric_value:.4f}")

        return saved_path

    def _get_worst_idx(self) -> int:
        """Get index of the worst checkpoint in saved list."""
        if self.mode == "min":
            return max(range(len(self.saved_checkpoints)), key=lambda i: self.saved_checkpoints[i][0])
        return min(range(len(self.saved_checkpoints)), key=lambda i: self.saved_checkpoints[i][0])


class LRLogger:
    """
    Logs learning rate at each epoch to console and optionally to a JSON file.

    Useful for debugging scheduler behavior and ensuring warmup works correctly.

    Args:
        log_file: Optional path to write LR history as JSON.
        verbose: Whether to log LR to console each epoch.
    """

    def __init__(self, log_file: Optional[str] = None, verbose: bool = True):
        self.log_file = Path(log_file) if log_file else None
        self.verbose = verbose
        self.lr_history: list = []

    def __call__(self, epoch: int, optimizer: torch.optim.Optimizer):
        """
        Log current learning rates.

        Args:
            epoch: Current epoch.
            optimizer: The optimizer whose param group LRs to log.
        """
        lrs = [pg["lr"] for pg in optimizer.param_groups]
        entry = {"epoch": epoch, "learning_rates": lrs}
        self.lr_history.append(entry)

        if self.verbose:
            lr_str = ", ".join(f"{lr:.2e}" for lr in lrs)
            logger.info(f"LRLogger: epoch {epoch} | lr=[{lr_str}]")

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "w") as f:
                json.dump(self.lr_history, f, indent=2)

    def get_history(self) -> list:
        """Return the full LR history."""
        return self.lr_history
