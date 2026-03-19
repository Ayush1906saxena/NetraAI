#!/usr/bin/env python3
"""
Training script for Diabetic Retinopathy grading.

Loads config, creates RETFoundDRGrader with LoRA, builds APTOS dataset
and data loaders, and runs the full training loop via DRTrainer.

Usage:
    python -m ml.scripts.train_dr
    python -m ml.scripts.train_dr --config ml/configs/dr_grading.yaml
    python -m ml.scripts.train_dr --resume outputs/dr_grading/checkpoints/last.pt
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.augmentations import get_train_transforms, get_val_transforms
from ml.data.dataset import FundusDataset
from ml.models.retfound_wrapper import RETFoundDRGrader
from ml.training.trainer import DRTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_model(config: dict) -> RETFoundDRGrader:
    """Instantiate the RETFoundDRGrader model from config."""
    model_cfg = config["model"]
    lora_cfg = model_cfg.get("lora", {})

    model = RETFoundDRGrader(
        num_classes=model_cfg.get("num_classes", 5),
        pretrained_path=model_cfg.get("pretrained_path", None),
        model_variant=model_cfg.get("model_variant", "mae"),
        lora_rank=lora_cfg.get("rank", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.1),
        use_lora=lora_cfg.get("enabled", True),
        drop_path=model_cfg.get("drop_path", 0.2),
    )

    return model


def build_dataloaders(config: dict) -> tuple:
    """
    Build train and validation DataLoaders.

    Returns:
        (train_loader, val_loader)
    """
    data_cfg = config["data"]
    train_cfg = config["training"]

    img_size = data_cfg.get("img_size", 224)
    data_root = data_cfg.get("data_root", "data/processed/aptos2019")
    batch_size = train_cfg.get("batch_size", 16)
    num_workers = data_cfg.get("num_workers", 4)
    pin_memory = data_cfg.get("pin_memory", True)
    use_weighted_sampler = train_cfg.get("use_weighted_sampler", True)

    # Transforms
    train_transform = get_train_transforms(img_size=img_size)
    val_transform = get_val_transforms(img_size=img_size)

    # Datasets
    train_dataset = FundusDataset(
        root=data_root,
        split="train",
        transform=train_transform,
        num_classes=data_cfg.get("num_classes", 5),
    )

    val_dataset = FundusDataset(
        root=data_root,
        split="val",
        transform=val_transform,
        num_classes=data_cfg.get("num_classes", 5),
    )

    logger.info(f"Train dataset: {train_dataset}")
    logger.info(f"Val dataset:   {val_dataset}")

    # Train loader with optional weighted sampling
    if use_weighted_sampler:
        sampler = train_dataset.get_weighted_sampler()
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # larger batch for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train DR grading model")
    parser.add_argument(
        "--config",
        type=str,
        default="ml/configs/dr_grading.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Set seed
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")

    # Check device
    device = config.get("training", {}).get("device", "mps")
    if device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS not available, falling back to CPU")
        config["training"]["device"] = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        config["training"]["device"] = "cpu"

    # Build model
    logger.info("Building model...")
    model = build_model(config)

    # Build data loaders
    logger.info("Building data loaders...")
    train_loader, val_loader = build_dataloaders(config)

    # Create trainer
    output_dir = config.get("output", {}).get("dir", "outputs/dr_grading")
    trainer = DRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir,
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    results = trainer.train()

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best QWK: {results['best_qwk']:.4f} at epoch {results['best_epoch']}")
    logger.info(f"Checkpoints saved to: {output_dir}/checkpoints/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
