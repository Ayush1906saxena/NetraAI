"""
DRTrainer: Full training loop for Diabetic Retinopathy grading.

Features:
  - AMP mixed precision on MPS (Apple Silicon)
  - LoRA fine-tuning support
  - Checkpointing with top-K and best model tracking
  - Early stopping
  - W&B and MLflow logging
  - Comprehensive validation metrics (QWK, AUC-ROC, sensitivity/specificity)
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from ml.training.callbacks import EarlyStopping, LRLogger, ModelCheckpoint
from ml.training.losses import LabelSmoothingCrossEntropy
from ml.training.metrics import (
    compute_auc,
    compute_ece,
    compute_qwk,
    compute_sensitivity_specificity,
)
from ml.training.schedulers import CosineWithWarmRestarts, WarmupCosineScheduler

logger = logging.getLogger(__name__)


class DRTrainer:
    """
    End-to-end trainer for DR grading models.

    Args:
        model: The neural network model (e.g., RETFoundDRGrader).
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        config: Dict with training configuration (from YAML).
        output_dir: Directory for checkpoints, logs, and artifacts.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        output_dir: str = "outputs/dr_grading",
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training config
        train_cfg = config.get("training", {})
        self.device = torch.device(train_cfg.get("device", "mps"))
        self.epochs = train_cfg.get("epochs", 50)
        self.grad_clip = train_cfg.get("grad_clip_norm", 1.0)
        self.use_amp = train_cfg.get("use_amp", True)
        self.num_classes = config.get("model", {}).get("num_classes", 5)

        # Model
        self.model = model.to(self.device)

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss
        label_smoothing = train_cfg.get("label_smoothing", 0.1)
        self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)

        # Optimizer
        lr = train_cfg.get("lr", 5e-4)
        weight_decay = train_cfg.get("weight_decay", 0.01)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Scheduler
        sched_cfg = config.get("scheduler", {})
        sched_type = sched_cfg.get("type", "warmup_cosine")
        warmup_epochs = sched_cfg.get("warmup_epochs", 5)
        min_lr = sched_cfg.get("min_lr", 1e-7)

        if sched_type == "cosine_warm_restarts":
            T_0 = sched_cfg.get("T_0", 10)
            T_mult = sched_cfg.get("T_mult", 2)
            self.scheduler = CosineWithWarmRestarts(
                self.optimizer,
                warmup_steps=warmup_epochs,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=min_lr,
            )
        else:
            self.scheduler = WarmupCosineScheduler(
                self.optimizer,
                warmup_steps=warmup_epochs,
                total_steps=self.epochs,
                min_lr=min_lr,
            )

        # AMP scaler - use CPU-based scaler for MPS compatibility
        if self.use_amp and self.device.type == "mps":
            # MPS supports float16 autocast but not GradScaler;
            # we still use autocast but skip scaling
            self.scaler = None
        elif self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Callbacks
        es_cfg = config.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 10),
            min_delta=es_cfg.get("min_delta", 1e-4),
            mode=es_cfg.get("mode", "max"),  # monitor QWK (higher is better)
        )

        self.checkpoint = ModelCheckpoint(
            save_dir=str(self.output_dir / "checkpoints"),
            monitor="val_qwk",
            mode="max",
            save_top_k=3,
            save_last=True,
        )

        self.lr_logger = LRLogger(
            log_file=str(self.output_dir / "lr_history.json"),
        )

        # Experiment tracking
        self.use_wandb = config.get("logging", {}).get("wandb", False)
        self.use_mlflow = config.get("logging", {}).get("mlflow", False)
        self.wandb_run = None
        self.mlflow_run = None

        self._init_tracking()

        # Best metrics tracking
        self.best_qwk = 0.0
        self.best_epoch = 0
        self.history: list = []

    def _init_tracking(self):
        """Initialize W&B and/or MLflow experiment tracking."""
        log_cfg = self.config.get("logging", {})

        if self.use_wandb:
            try:
                import wandb

                project = log_cfg.get("wandb_project", "netra-dr-grading")
                run_name = log_cfg.get("run_name", None)
                self.wandb_run = wandb.init(
                    project=project,
                    name=run_name,
                    config=self.config,
                    dir=str(self.output_dir),
                )
                logger.info(f"W&B initialized: {self.wandb_run.url}")
            except ImportError:
                logger.warning("wandb not installed, skipping W&B logging")
                self.use_wandb = False

        if self.use_mlflow:
            try:
                import mlflow

                experiment_name = log_cfg.get("mlflow_experiment", "dr-grading")
                tracking_uri = log_cfg.get("mlflow_tracking_uri", "mlruns")
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(experiment_name)
                self.mlflow_run = mlflow.start_run(
                    run_name=log_cfg.get("run_name", None)
                )
                mlflow.log_params(
                    {
                        "model": self.config.get("model", {}).get("name", "unknown"),
                        "epochs": self.epochs,
                        "lr": self.config.get("training", {}).get("lr", 0),
                        "batch_size": self.config.get("training", {}).get("batch_size", 0),
                    }
                )
                logger.info(f"MLflow run started: {self.mlflow_run.info.run_id}")
            except ImportError:
                logger.warning("mlflow not installed, skipping MLflow logging")
                self.use_mlflow = False

    def _log_metrics(self, metrics: dict, step: int):
        """Log metrics to all enabled trackers."""
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

        if self.use_mlflow:
            import mlflow
            mlflow.log_metrics(metrics, step=step)

    def train(self) -> dict:
        """
        Run the full training loop.

        Returns:
            Dict with best metrics and training history.
        """
        logger.info(f"Starting training for {self.epochs} epochs on {self.device}")
        logger.info(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        logger.info(f"Output directory: {self.output_dir}")

        for epoch in range(self.epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self._train_epoch(epoch)

            # Validate
            val_metrics = self._validate(epoch)

            # Step scheduler
            self.scheduler.step()

            # Log LR
            self.lr_logger(epoch, self.optimizer)

            # Combine metrics
            all_metrics = {}
            all_metrics.update({f"train_{k}": v for k, v in train_metrics.items()})
            all_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            all_metrics["epoch"] = epoch
            all_metrics["epoch_time"] = time.time() - epoch_start

            self._log_metrics(all_metrics, step=epoch)
            self.history.append(all_metrics)

            # Log summary
            logger.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val QWK: {val_metrics['qwk']:.4f} | "
                f"Val AUC: {val_metrics['auc']:.4f} | "
                f"Sens@90: {val_metrics['sensitivity']:.4f} | "
                f"Spec@90: {val_metrics['specificity']:.4f} | "
                f"Time: {all_metrics['epoch_time']:.1f}s"
            )

            # Checkpoint
            val_qwk = val_metrics["qwk"]
            self._save_checkpoint(epoch, val_qwk, val_metrics)

            # Track best
            if val_qwk > self.best_qwk:
                self.best_qwk = val_qwk
                self.best_epoch = epoch

            # Early stopping (monitor QWK)
            if self.early_stopping(epoch, val_qwk):
                logger.info(
                    f"Early stopping at epoch {epoch}. "
                    f"Best QWK: {self.best_qwk:.4f} at epoch {self.best_epoch}"
                )
                break

        # Finish tracking
        if self.use_wandb:
            import wandb
            wandb.finish()

        if self.use_mlflow:
            import mlflow
            mlflow.end_run()

        return {
            "best_qwk": self.best_qwk,
            "best_epoch": self.best_epoch,
            "history": self.history,
        }

    def _train_epoch(self, epoch: int) -> dict:
        """
        Run one training epoch.

        Args:
            epoch: Current epoch index.

        Returns:
            Dict with training metrics: loss, accuracy.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                # Use autocast for MPS or CUDA
                device_type = "cpu" if self.device.type == "mps" else self.device.type
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                if self.scaler is not None:
                    # CUDA path with GradScaler
                    self.scaler.scale(loss).backward()
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # MPS path: autocast without scaler
                    loss.backward()
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        avg_loss = running_loss / total
        accuracy = correct / total

        return {"loss": avg_loss, "accuracy": accuracy}

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict:
        """
        Run validation and compute comprehensive metrics.

        Args:
            epoch: Current epoch index.

        Returns:
            Dict with: loss, accuracy, qwk, auc, sensitivity, specificity,
            threshold, youden_j, ece.
        """
        self.model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.use_amp:
                device_type = "cpu" if self.device.type == "mps" else self.device.type
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.softmax(logits.float(), dim=1)
            preds = probs.argmax(dim=1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)
        total = len(all_labels)

        avg_loss = running_loss / total
        accuracy = (all_preds == all_labels).mean()

        # Quadratic Weighted Kappa
        qwk = compute_qwk(all_labels, all_preds)

        # AUC-ROC for referable DR (grade >= 2)
        try:
            auc = compute_auc(all_labels, all_probs, referable_threshold=2)
        except ValueError:
            auc = 0.0
            logger.warning("AUC computation failed (possibly only one class in batch)")

        # Sensitivity / Specificity at 90% sensitivity
        try:
            sens_spec = compute_sensitivity_specificity(
                all_labels, all_probs, target_sensitivity=0.90, referable_threshold=2
            )
        except ValueError:
            sens_spec = {"sensitivity": 0.0, "specificity": 0.0, "threshold": 0.5, "youden_j": 0.0}
            logger.warning("Sensitivity/specificity computation failed")

        # Expected Calibration Error
        ece = compute_ece(all_labels, all_probs)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "qwk": qwk,
            "auc": auc,
            "sensitivity": sens_spec["sensitivity"],
            "specificity": sens_spec["specificity"],
            "threshold": sens_spec["threshold"],
            "youden_j": sens_spec["youden_j"],
            "ece": ece,
        }

    def _save_checkpoint(
        self,
        epoch: int,
        val_qwk: float,
        val_metrics: dict,
    ) -> Optional[Path]:
        """
        Save checkpoint via the ModelCheckpoint callback.

        Args:
            epoch: Current epoch.
            val_qwk: QWK score (monitored metric).
            val_metrics: Full validation metrics dict for extra_info.

        Returns:
            Path to saved checkpoint if one was saved, else None.
        """
        return self.checkpoint(
            epoch=epoch,
            metric_value=val_qwk,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            extra_info=val_metrics,
        )

    def load_checkpoint(self, checkpoint_path: str):
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint .pt file.
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        epoch = ckpt.get("epoch", 0)
        logger.info(f"Resumed from checkpoint at epoch {epoch}")
        return epoch
