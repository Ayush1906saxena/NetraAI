"""
Simplified inference service using the trained EfficientNet-B3 DR model.

Loads the actual trained checkpoint from checkpoints/dr_aptos/best.pth
and provides analyze_fundus() and check_quality() methods.

Enhanced with:
- Async wrappers that run inference in a thread pool
- Redis caching for prediction results (keyed by image SHA-256)
- Inference request counting and latency metrics
- Model warm-up on load (dummy inference to pre-compile)
- Test-Time Augmentation (TTA) for improved accuracy
- Temperature scaling / confidence calibration
- Multi-model inference (DR + IQA + Glaucoma)
- Monte Carlo Dropout for uncertainty estimation
- Ensemble support from multiple DR checkpoints
"""

import asyncio
import base64
import hashlib
import io
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── Input validation constants ────────────────────────────────────────────
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB
MIN_IMAGE_SIZE_BYTES = 1024              # 1 KB (likely corrupt below this)
ALLOWED_IMAGE_MAGIC = {
    b"\xff\xd8\xff": "JPEG",
    b"\x89PNG": "PNG",
    b"BM": "BMP",
    b"GIF8": "GIF",
    b"RIFF": "WEBP",
}


def _validate_image_input(image_bytes: bytes) -> None:
    """
    Validate raw image bytes before processing.

    Raises:
        ValueError: If the image fails validation checks.
    """
    if not isinstance(image_bytes, bytes):
        raise ValueError("Image input must be bytes.")

    size = len(image_bytes)
    if size < MIN_IMAGE_SIZE_BYTES:
        raise ValueError(
            f"Image too small ({size} bytes). Minimum is {MIN_IMAGE_SIZE_BYTES} bytes. "
            "The file may be corrupt or empty."
        )
    if size > MAX_IMAGE_SIZE_BYTES:
        raise ValueError(
            f"Image too large ({size / (1024 * 1024):.1f} MB). "
            f"Maximum allowed is {MAX_IMAGE_SIZE_BYTES / (1024 * 1024):.0f} MB."
        )

    # Check magic bytes for known image formats
    detected = False
    for magic, fmt in ALLOWED_IMAGE_MAGIC.items():
        if image_bytes[:len(magic)] == magic:
            detected = True
            break
    if not detected:
        raise ValueError(
            "Unsupported image format. Please upload a JPEG, PNG, BMP, GIF, or WEBP image."
        )

# DR grade labels and clinical info
DR_GRADE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
DR_GRADE_DESCRIPTIONS = [
    "No diabetic retinopathy detected.",
    "Mild non-proliferative diabetic retinopathy — microaneurysms only.",
    "Moderate non-proliferative diabetic retinopathy — more than just microaneurysms.",
    "Severe non-proliferative diabetic retinopathy — extensive hemorrhages, venous beading.",
    "Proliferative diabetic retinopathy — neovascularization or vitreous/preretinal hemorrhage.",
]
REFERRAL_THRESHOLD = 2  # grade >= 2 is referable DR


# ── Inference metrics (simple in-memory dict) ─────────────────────────────
inference_metrics: dict[str, Any] = {
    "total_requests": 0,
    "total_errors": 0,
    "total_cache_hits": 0,
    "total_cache_misses": 0,
    "total_latency_ms": 0.0,
    "avg_latency_ms": 0.0,
    "quality_checks": 0,
    "tta_requests": 0,
    "uncertainty_requests": 0,
    "ensemble_requests": 0,
    "iqa_requests": 0,
    "glaucoma_requests": 0,
}


class DRGrader(nn.Module):
    """EfficientNet-B3 DR Grader — must match the architecture used during training."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

        backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1536, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)


def _enable_dropout(model: nn.Module) -> None:
    """Enable dropout layers during inference for MC Dropout."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


def _disable_dropout(model: nn.Module) -> None:
    """Restore eval mode on dropout layers."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.eval()


class InferenceService:
    """
    Loads the trained EfficientNet-B3 checkpoint and runs DR inference
    with preprocessing, GradCAM, and quality checks.

    Enhanced with:
    - Multi-model support (DR + IQA + Glaucoma)
    - Test-Time Augmentation (TTA)
    - Temperature scaling / confidence calibration
    - Monte Carlo Dropout uncertainty estimation
    - Ensemble support from multiple DR checkpoints
    """

    def __init__(self):
        self.model: Optional[DRGrader] = None
        self.device: str = "cpu"
        self.is_loaded: bool = False
        self._preprocessor = None
        self._lock = asyncio.Lock()

        # Multi-model support
        self.iqa_model = None
        self.iqa_loaded: bool = False
        self.iqa_checkpoint_path: Optional[str] = None

        self.glaucoma_model = None
        self.glaucoma_loaded: bool = False
        self.glaucoma_checkpoint_path: Optional[str] = None

        # Ensemble support
        self.ensemble_models: list[DRGrader] = []
        self.ensemble_checkpoint_paths: list[str] = []
        self.ensemble_size: int = 0

        # Temperature scaling
        self.temperature_scaler = None
        self.temperature_value: float = 1.5  # default

        # Checkpoint paths for model info
        self.dr_checkpoint_path: Optional[str] = None

        # Settings cache
        self._settings_loaded = False
        self._tta_enabled = False
        self._tta_n_augments = 5
        self._mc_dropout_passes = 5
        self._iqa_quality_threshold = 0.4
        self._uncertainty_review_threshold = 0.15
        self._ensemble_max = 3

    def _load_settings(self) -> None:
        """Load settings from config (lazy, once)."""
        if self._settings_loaded:
            return
        try:
            from server.config import settings
            self._tta_enabled = settings.TTA_ENABLED
            self._tta_n_augments = settings.TTA_N_AUGMENTS
            self._mc_dropout_passes = settings.MC_DROPOUT_PASSES
            self._iqa_quality_threshold = settings.IQA_QUALITY_THRESHOLD
            self._uncertainty_review_threshold = settings.UNCERTAINTY_REVIEW_THRESHOLD
            self._ensemble_max = settings.ENSEMBLE_MODELS
            self.temperature_value = settings.TEMPERATURE_SCALING_DEFAULT
        except Exception as e:
            logger.warning("Could not load settings, using defaults: %s", e)
        self._settings_loaded = True

    async def load_model(self, checkpoint_path: str, device: str = "cpu") -> None:
        """Load the trained DR model from a checkpoint file."""
        async with self._lock:
            if self.is_loaded:
                logger.info("Model already loaded, skipping.")
                return

            ckpt_path = Path(checkpoint_path)
            if not ckpt_path.exists():
                logger.error("Checkpoint not found: %s", checkpoint_path)
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            self.device = device
            self.dr_checkpoint_path = checkpoint_path
            logger.info("Loading DR model from %s on device=%s ...", checkpoint_path, device)

            # Load model in a thread to avoid blocking the event loop
            def _load():
                model = DRGrader(num_classes=5)
                checkpoint = torch.load(
                    str(ckpt_path),
                    map_location=device,
                    weights_only=False,
                )
                # The best.pth has model_state_dict key; epoch checkpoints have raw state_dict
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                    metrics = checkpoint.get("metrics", {})
                    epoch = checkpoint.get("epoch", "?")
                    best_qwk = checkpoint.get("best_qwk", "?")
                    logger.info(
                        "Loaded best checkpoint from epoch %s (QWK=%.4f, metrics=%s)",
                        epoch, best_qwk if isinstance(best_qwk, float) else 0, metrics,
                    )
                else:
                    model.load_state_dict(checkpoint)
                    logger.info("Loaded raw state_dict checkpoint.")

                model.to(device)
                model.eval()
                return model

            self.model = await asyncio.to_thread(_load)

            # Initialize the preprocessor
            from ml.data.preprocess import FundusPreprocessor
            self._preprocessor = FundusPreprocessor()

            self.is_loaded = True
            self._load_settings()
            logger.info("DR model loaded successfully. Ready for inference.")

            # Warm-up: run a dummy inference to pre-compile model graph
            await self._warmup()

    async def load_iqa_model(self, checkpoint_path: str) -> None:
        """Load the FundusIQA model from a checkpoint file."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            logger.warning("IQA checkpoint not found: %s — skipping IQA model.", checkpoint_path)
            return

        def _load():
            from ml.models.iqa_model import FundusIQA
            model = FundusIQA()
            state_dict = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
            # Handle both raw state_dict and wrapped checkpoint
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"])
            elif isinstance(state_dict, dict) and any(k.startswith("features.") for k in state_dict):
                model.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            return model

        try:
            self.iqa_model = await asyncio.to_thread(_load)
            self.iqa_loaded = True
            self.iqa_checkpoint_path = checkpoint_path
            logger.info("IQA model loaded from %s", checkpoint_path)
        except Exception as e:
            logger.warning("Failed to load IQA model: %s — skipping.", e)
            self.iqa_model = None
            self.iqa_loaded = False

    async def load_glaucoma_model(self, checkpoint_path: str) -> None:
        """Load the GlaucomaSegmentor model from a checkpoint file."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            logger.warning(
                "Glaucoma checkpoint not found: %s — skipping glaucoma model.", checkpoint_path
            )
            return

        def _load():
            from ml.models.glaucoma_unet import GlaucomaSegmentor
            model = GlaucomaSegmentor()
            state_dict = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"])
            elif isinstance(state_dict, dict) and any(k.startswith("unet.") for k in state_dict):
                model.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            return model

        try:
            self.glaucoma_model = await asyncio.to_thread(_load)
            self.glaucoma_loaded = True
            self.glaucoma_checkpoint_path = checkpoint_path
            logger.info("Glaucoma model loaded from %s", checkpoint_path)
        except Exception as e:
            logger.warning("Failed to load glaucoma model: %s — skipping.", e)
            self.glaucoma_model = None
            self.glaucoma_loaded = False

    async def load_ensemble_models(self, checkpoints_dir: str) -> None:
        """
        Load multiple DR model checkpoints for ensemble prediction.

        Scans checkpoints_dir for .pth files (excluding best.pth which is already
        loaded as the primary model), sorts by modification time, and loads
        up to ENSEMBLE_MODELS checkpoints.
        """
        self._load_settings()
        ckpt_dir = Path(checkpoints_dir)
        if not ckpt_dir.exists():
            logger.warning("Ensemble directory not found: %s — skipping.", checkpoints_dir)
            return

        # Find all .pth files except best.pth (already loaded as primary)
        pth_files = sorted(
            [f for f in ckpt_dir.glob("*.pth") if f.name != "best.pth"],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

        if not pth_files:
            logger.info("No additional checkpoints found for ensemble in %s.", checkpoints_dir)
            return

        # Load top-N models
        max_ensemble = self._ensemble_max
        to_load = pth_files[:max_ensemble]

        def _load_one(path: Path) -> Optional[DRGrader]:
            try:
                model = DRGrader(num_classes=5)
                checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                model.to(self.device)
                model.eval()
                logger.info("Ensemble: loaded %s", path.name)
                return model
            except Exception as e:
                logger.warning("Ensemble: failed to load %s: %s", path.name, e)
                return None

        def _load_all():
            models = []
            paths = []
            for p in to_load:
                m = _load_one(p)
                if m is not None:
                    models.append(m)
                    paths.append(str(p))
            return models, paths

        self.ensemble_models, self.ensemble_checkpoint_paths = await asyncio.to_thread(_load_all)
        self.ensemble_size = len(self.ensemble_models)
        if self.ensemble_size > 0:
            logger.info(
                "Ensemble loaded: %d additional models from %s", self.ensemble_size, checkpoints_dir
            )
        else:
            logger.info("No ensemble models were successfully loaded.")

    async def load_temperature_scaling(self, calibration_path: str) -> None:
        """Load a saved temperature scaling parameter from calibration checkpoint."""
        cal_path = Path(calibration_path)
        if not cal_path.exists():
            logger.info(
                "No calibration checkpoint at %s — using default temperature %.2f",
                calibration_path,
                self.temperature_value,
            )
            return

        def _load():
            from ml.evaluation.calibration import TemperatureScaling
            ts = TemperatureScaling.load(str(cal_path))
            return ts

        try:
            self.temperature_scaler = await asyncio.to_thread(_load)
            self.temperature_value = self.temperature_scaler.temperature.item()
            logger.info(
                "Temperature scaling loaded from %s (T=%.4f)", calibration_path, self.temperature_value
            )
        except Exception as e:
            logger.warning("Failed to load temperature scaling: %s — using default.", e)

    async def _warmup(self) -> None:
        """Run a dummy inference to pre-compile the model (avoids cold-start latency)."""
        if not self.is_loaded or self.model is None:
            return

        def _run_dummy():
            dummy = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy)

        try:
            await asyncio.to_thread(_run_dummy)
            logger.info("Model warm-up complete (dummy inference executed).")
        except Exception as exc:
            logger.warning("Model warm-up failed (non-fatal): %s", exc)

    async def unload_model(self) -> None:
        """Release all models from memory."""
        async with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
                self.is_loaded = False

            if self.iqa_model is not None:
                del self.iqa_model
                self.iqa_model = None
                self.iqa_loaded = False

            if self.glaucoma_model is not None:
                del self.glaucoma_model
                self.glaucoma_model = None
                self.glaucoma_loaded = False

            for m in self.ensemble_models:
                del m
            self.ensemble_models = []
            self.ensemble_size = 0

            self.temperature_scaler = None

            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("All models unloaded.")

    def _preprocess(self, image_bytes: bytes) -> tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess raw image bytes into a model-ready tensor and the
        preprocessed RGB image (for GradCAM overlay).

        Returns:
            (tensor [1,3,224,224], preprocessed_rgb [H,W,3] uint8)
        """
        from ml.data.augmentations import IMAGENET_MEAN, IMAGENET_STD

        # Use FundusPreprocessor: circle crop, Ben Graham, CLAHE, resize to 224
        processed = self._preprocessor.process_bytes(image_bytes, target_size=224)
        # processed is (224, 224, 3) uint8 RGB

        # Keep a copy for GradCAM overlay
        processed_rgb = processed.copy()

        # Normalize for model input
        img_float = processed.astype(np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        img_float = (img_float - mean) / std

        # HWC -> CHW -> NCHW
        tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor, processed_rgb

    def _generate_gradcam_bytes(
        self, input_tensor: torch.Tensor, original_image: np.ndarray, target_class: int
    ) -> bytes:
        """Generate GradCAM overlay PNG bytes."""
        from ml.evaluation.gradcam import generate_gradcam

        return generate_gradcam(
            model=self.model,
            input_tensor=input_tensor,
            original_image=original_image,
            target_class=target_class,
        )

    def _apply_temperature_scaling(self, logits: torch.Tensor) -> np.ndarray:
        """Apply temperature scaling to logits and return calibrated probabilities."""
        if self.temperature_scaler is not None:
            calibrated = self.temperature_scaler.calibrate(logits)
            return calibrated.cpu().numpy()[0]
        else:
            # Use default temperature
            scaled = logits / self.temperature_value
            return F.softmax(scaled, dim=1).cpu().numpy()[0]

    def _run_tta(self, input_tensor: torch.Tensor, n_augments: int = 5) -> np.ndarray:
        """
        Run Test-Time Augmentation on the input tensor.

        Applies augmentations: original, hflip, vflip, rotate90, rotate90+hflip
        and averages softmax probabilities across all augments.

        Args:
            input_tensor: (1, 3, H, W) preprocessed tensor.
            n_augments: Number of augmentations to use (1-5).

        Returns:
            (C,) averaged softmax probabilities.
        """
        input_t = input_tensor.to(self.device)

        # Define 5 augmentations
        augmentations = [
            ("original", lambda x: x),
            ("hflip", lambda x: x.flip(-1)),
            ("vflip", lambda x: x.flip(-2)),
            ("rot90", lambda x: torch.rot90(x, k=1, dims=[-2, -1])),
            ("rot90_hflip", lambda x: torch.rot90(x, k=1, dims=[-2, -1]).flip(-1)),
        ]
        augmentations = augmentations[:n_augments]

        all_probs = []
        with torch.no_grad():
            for _name, aug_fn in augmentations:
                augmented = aug_fn(input_t)
                logits = self.model(augmented)
                probs = F.softmax(logits, dim=1).cpu()
                all_probs.append(probs)

        avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
        return avg_probs.squeeze(0).numpy()

    def _run_mc_dropout(self, input_tensor: torch.Tensor, n_passes: int = 5) -> dict[str, Any]:
        """
        Run Monte Carlo Dropout for uncertainty estimation.

        Runs inference N times with dropout enabled to estimate predictive
        uncertainty. High standard deviation indicates the model is uncertain.

        Args:
            input_tensor: (1, 3, H, W) preprocessed tensor.
            n_passes: Number of forward passes with dropout enabled.

        Returns:
            dict with mean_probs, std_probs, uncertainty_score, needs_review.
        """
        input_t = input_tensor.to(self.device)

        # Enable dropout layers while keeping batchnorm in eval
        _enable_dropout(self.model)

        all_probs = []
        with torch.no_grad():
            for _ in range(n_passes):
                logits = self.model(input_t)
                probs = F.softmax(logits, dim=1).cpu()
                all_probs.append(probs)

        # Restore eval mode
        _disable_dropout(self.model)

        stacked = torch.stack(all_probs, dim=0)  # (N, 1, C)
        mean_probs = stacked.mean(dim=0).squeeze(0).numpy()  # (C,)
        std_probs = stacked.std(dim=0).squeeze(0).numpy()    # (C,)

        uncertainty_score = float(std_probs.mean())

        return {
            "mean_probs": mean_probs,
            "std_probs": std_probs,
            "uncertainty_score": round(uncertainty_score, 4),
            "needs_review": uncertainty_score > self._uncertainty_review_threshold,
        }

    def _run_ensemble(self, input_tensor: torch.Tensor) -> Optional[np.ndarray]:
        """
        Run ensemble inference using loaded additional DR models.

        Averages predictions from the primary model + ensemble models.

        Args:
            input_tensor: (1, 3, H, W) preprocessed tensor.

        Returns:
            (C,) averaged softmax probabilities, or None if no ensemble models.
        """
        if not self.ensemble_models:
            return None

        input_t = input_tensor.to(self.device)
        all_probs = []

        with torch.no_grad():
            # Primary model prediction
            logits = self.model(input_t)
            probs = F.softmax(logits, dim=1).cpu()
            all_probs.append(probs)

            # Ensemble model predictions
            for ens_model in self.ensemble_models:
                logits = ens_model(input_t)
                probs = F.softmax(logits, dim=1).cpu()
                all_probs.append(probs)

        avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
        return avg_probs.squeeze(0).numpy()

    def _run_iqa(self, input_tensor: torch.Tensor) -> Optional[dict[str, Any]]:
        """
        Run Image Quality Assessment using the IQA model.

        Args:
            input_tensor: (1, 3, H, W) preprocessed tensor.

        Returns:
            dict with quality_score, is_gradeable, issues, accept or None.
        """
        if not self.iqa_loaded or self.iqa_model is None:
            return None

        input_t = input_tensor.to(self.device)
        try:
            result = self.iqa_model.assess(input_t)
            return result
        except Exception as e:
            logger.warning("IQA inference failed: %s", e)
            return None

    def _run_glaucoma(self, input_tensor: torch.Tensor) -> Optional[dict[str, Any]]:
        """
        Run glaucoma screening using the GlaucomaSegmentor model.

        Note: Glaucoma model expects 512x512 input, so we resize the 224x224 tensor.

        Args:
            input_tensor: (1, 3, H, W) preprocessed tensor.

        Returns:
            dict with vertical_cdr, area_cdr, risk_level, disc_area_px, cup_area_px or None.
        """
        if not self.glaucoma_loaded or self.glaucoma_model is None:
            return None

        # Resize tensor from 224x224 to 512x512 for glaucoma model
        input_t = input_tensor.to(self.device)
        input_resized = F.interpolate(input_t, size=(512, 512), mode="bilinear", align_corners=False)

        try:
            results = self.glaucoma_model.assess_glaucoma(input_resized)
            return results[0] if results else None
        except Exception as e:
            logger.warning("Glaucoma inference failed: %s", e)
            return None

    async def analyze_fundus(
        self,
        image_bytes: bytes,
        use_tta: Optional[bool] = None,
        use_uncertainty: bool = True,
        use_ensemble: bool = True,
        high_confidence_mode: bool = False,
    ) -> dict[str, Any]:
        """
        Full multi-model analysis pipeline:
        1. Validate input image
        2. Preprocess (circle crop, Ben Graham, CLAHE, normalize)
        3. Run IQA model (if loaded) — return early if quality too low
        4. Run DR grading (with optional TTA and ensemble)
        5. Apply temperature scaling for calibrated confidence
        6. Run Monte Carlo Dropout for uncertainty estimation
        7. Run Glaucoma segmentation (if loaded)
        8. Generate GradCAM heatmap
        9. Compute referral recommendation

        Args:
            image_bytes: Raw image file bytes (JPEG/PNG)
            use_tta: Override TTA setting (None = use config default)
            use_uncertainty: Whether to run MC Dropout uncertainty estimation
            use_ensemble: Whether to use ensemble models if available
            high_confidence_mode: Enable TTA + ensemble for maximum accuracy

        Returns:
            dict with dr, iqa, glaucoma, uncertainty, referral, gradcam, model_info, meta

        Raises:
            ValueError: If image fails validation (bad format, size).
            RuntimeError: If model is not loaded.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self._load_settings()
        t_start = time.perf_counter()

        # Resolve TTA flag
        if high_confidence_mode:
            use_tta = True
            use_ensemble = True
            use_uncertainty = True
        if use_tta is None:
            use_tta = self._tta_enabled

        # ── Input validation ──────────────────────────────────────────
        _validate_image_input(image_bytes)

        # ── Preprocess ────────────────────────────────────────────────
        try:
            input_tensor, processed_rgb = await asyncio.to_thread(
                self._preprocess, image_bytes
            )
        except Exception as e:
            logger.error("Preprocessing failed: %s", e, exc_info=True)
            raise ValueError(
                f"Failed to preprocess image: {e}. "
                "Ensure the file is a valid retinal fundus photograph."
            ) from e

        # ── Track which models ran ────────────────────────────────────
        models_used = []
        result = {}

        # ── IQA (run first — gate for further analysis) ───────────────
        iqa_result = None
        if self.iqa_loaded:
            def _iqa():
                return self._run_iqa(input_tensor)
            iqa_result = await asyncio.to_thread(_iqa)
            inference_metrics["iqa_requests"] += 1
            models_used.append("iqa")

            if iqa_result is not None:
                result["iqa"] = {
                    "quality_score": iqa_result.get("quality_score"),
                    "is_gradeable": iqa_result.get("is_gradeable", True),
                    "issues": iqa_result.get("issues", []),
                    "guidance": (
                        "Image quality is too low for reliable analysis. "
                        "Please retake with better lighting and focus."
                        if not iqa_result.get("accept", True)
                        else None
                    ),
                }

                # Gate: if quality too low, return early
                q_score = iqa_result.get("quality_score", 1.0)
                if q_score < self._iqa_quality_threshold and not iqa_result.get("accept", True):
                    total_time_ms = (time.perf_counter() - t_start) * 1000
                    return {
                        "status": "quality_rejected",
                        "analysis": {
                            "iqa": result["iqa"],
                            "dr": None,
                            "glaucoma": None,
                            "uncertainty": None,
                            "tta_used": False,
                            "models_used": models_used,
                            "inference_time_ms": round(total_time_ms, 1),
                        },
                        "referral": {
                            "is_referable": False,
                            "urgency": "none",
                            "risk_level": "unknown",
                            "recommendation": "Image quality insufficient. Please retake the image.",
                        },
                        "gradcam": {"overlay_png_base64": "", "has_gradcam": False},
                        "model_info": self._get_model_info(),
                        "meta": self._get_meta(),
                    }

        # ── DR Inference ──────────────────────────────────────────────
        def _infer_dr():
            t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    input_t = input_tensor.to(self.device)
                    logits = self.model(input_t)
                    raw_probs = F.softmax(logits, dim=1).cpu().numpy()[0]

                    # Calibrated confidence via temperature scaling
                    calibrated_probs = self._apply_temperature_scaling(logits)

                # TTA
                tta_probs = None
                if use_tta:
                    tta_probs = self._run_tta(input_tensor, n_augments=self._tta_n_augments)

                # Ensemble
                ensemble_probs = None
                if use_ensemble and self.ensemble_size > 0:
                    ensemble_probs = self._run_ensemble(input_tensor)

                # Combine: if TTA or ensemble is used, prefer their averaged results
                if tta_probs is not None and ensemble_probs is not None:
                    # Average TTA and ensemble results with raw probs
                    final_probs = (raw_probs + tta_probs + ensemble_probs) / 3.0
                elif tta_probs is not None:
                    final_probs = (raw_probs + tta_probs) / 2.0
                elif ensemble_probs is not None:
                    final_probs = (raw_probs + ensemble_probs) / 2.0
                else:
                    final_probs = raw_probs

            except Exception as exc:
                logger.error("Model inference failed: %s", exc, exc_info=True)
                raise RuntimeError(
                    "Model inference failed. The image may be incompatible or "
                    "the model encountered a numerical error."
                ) from exc

            latency_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "DR inference completed in %.1fms on %s (tta=%s, ensemble=%s)",
                latency_ms, self.device, use_tta, self.ensemble_size > 0,
            )
            return raw_probs, calibrated_probs, final_probs, tta_probs, ensemble_probs, latency_ms

        (
            raw_probs, calibrated_probs, final_probs,
            tta_probs, ensemble_probs, dr_latency_ms,
        ) = await asyncio.to_thread(_infer_dr)

        models_used.append("dr_efficientnet_b3")
        if use_tta:
            inference_metrics["tta_requests"] += 1
        if ensemble_probs is not None:
            inference_metrics["ensemble_requests"] += 1

        # Update metrics
        inference_metrics["total_requests"] += 1
        inference_metrics["total_latency_ms"] += dr_latency_ms
        inference_metrics["avg_latency_ms"] = (
            inference_metrics["total_latency_ms"] / inference_metrics["total_requests"]
        )

        grade = int(np.argmax(final_probs))
        raw_confidence = float(raw_probs[grade])
        calibrated_confidence = float(calibrated_probs[grade])
        final_confidence = float(final_probs[grade])

        result["dr"] = {
            "grade": grade,
            "grade_name": DR_GRADE_NAMES[grade],
            "description": DR_GRADE_DESCRIPTIONS[grade],
            "confidence": round(final_confidence, 4),
            "raw_confidence": round(raw_confidence, 4),
            "calibrated_confidence": round(calibrated_confidence, 4),
            "probabilities": {
                DR_GRADE_NAMES[i]: round(float(final_probs[i]), 4) for i in range(5)
            },
            "probabilities_list": [round(float(p), 4) for p in final_probs],
        }

        # ── Uncertainty Estimation (MC Dropout) ───────────────────────
        uncertainty_result = None
        if use_uncertainty:
            def _mc():
                return self._run_mc_dropout(input_tensor, n_passes=self._mc_dropout_passes)
            uncertainty_result = await asyncio.to_thread(_mc)
            inference_metrics["uncertainty_requests"] += 1

            result["uncertainty"] = {
                "score": uncertainty_result["uncertainty_score"],
                "needs_review": uncertainty_result["needs_review"],
                "method": "mc_dropout",
                "n_passes": self._mc_dropout_passes,
                "std_per_class": {
                    DR_GRADE_NAMES[i]: round(float(uncertainty_result["std_probs"][i]), 4)
                    for i in range(5)
                },
            }
        else:
            result["uncertainty"] = None

        # ── Glaucoma Segmentation ─────────────────────────────────────
        if self.glaucoma_loaded:
            def _glaucoma():
                return self._run_glaucoma(input_tensor)
            glaucoma_result = await asyncio.to_thread(_glaucoma)
            inference_metrics["glaucoma_requests"] += 1
            models_used.append("glaucoma_unet")

            if glaucoma_result is not None:
                result["glaucoma"] = {
                    "cdr": glaucoma_result.get("vertical_cdr"),
                    "area_cdr": glaucoma_result.get("area_cdr"),
                    "disc_area": glaucoma_result.get("disc_area_px", 0),
                    "cup_area": glaucoma_result.get("cup_area_px", 0),
                    "risk": glaucoma_result.get("risk_level", "unknown"),
                }
            else:
                result["glaucoma"] = None
        else:
            result["glaucoma"] = None

        # ── GradCAM ──────────────────────────────────────────────────
        try:
            gradcam_bytes = await asyncio.to_thread(
                self._generate_gradcam_bytes,
                input_tensor,
                processed_rgb,
                grade,
            )
            gradcam_b64 = base64.b64encode(gradcam_bytes).decode("utf-8")
        except Exception as e:
            logger.warning("GradCAM generation failed: %s", e)
            gradcam_bytes = b""
            gradcam_b64 = ""

        # ── Referral logic ────────────────────────────────────────────
        is_referable = grade >= REFERRAL_THRESHOLD
        if grade >= 4:
            urgency = "emergency"
            risk = "critical"
        elif grade == 3:
            urgency = "urgent"
            risk = "high"
        elif grade == 2:
            urgency = "soon"
            risk = "moderate"
        elif grade == 1:
            urgency = "routine"
            risk = "low"
        else:
            urgency = "none"
            risk = "minimal"

        # Flag if uncertain
        referral_note = ""
        if uncertainty_result and uncertainty_result["needs_review"]:
            referral_note = " (Note: model uncertainty is high — human review recommended.)"

        # ── Finalize TTA / ensemble metadata ──────────────────────────
        result["tta_used"] = use_tta
        result["models_used"] = models_used

        total_time_ms = (time.perf_counter() - t_start) * 1000
        result["inference_time_ms"] = round(total_time_ms, 1)

        # Fill in IQA if not already set
        if "iqa" not in result:
            result["iqa"] = None

        return {
            "status": "success",
            "analysis": result,
            "referral": {
                "is_referable": is_referable,
                "urgency": urgency,
                "risk_level": risk,
                "recommendation": (
                    f"Refer to ophthalmologist ({urgency}): {DR_GRADE_NAMES[grade]} detected.{referral_note}"
                    if is_referable
                    else f"No immediate referral needed. Routine follow-up in 12 months.{referral_note}"
                ),
            },
            "gradcam": {
                "overlay_png_base64": gradcam_b64,
                "has_gradcam": bool(gradcam_b64),
            },
            "model_info": self._get_model_info(),
            "meta": self._get_meta(),
        }

    async def analyze_fundus_with_tta(
        self, image_bytes: bytes, n_augments: int = 5
    ) -> dict[str, Any]:
        """
        Convenience method: run full analysis with TTA enabled.

        Applies 5 augmentations (original, hflip, vflip, rotate90, rotate90+hflip),
        averages softmax probabilities for improved accuracy (typically 1-3% improvement).

        Args:
            image_bytes: Raw image file bytes (JPEG/PNG).
            n_augments: Number of augmentations (1-5). Default 5.

        Returns:
            Full analysis result dict with TTA applied.
        """
        # Temporarily override n_augments
        orig = self._tta_n_augments
        self._tta_n_augments = min(n_augments, 5)
        try:
            return await self.analyze_fundus(image_bytes, use_tta=True)
        finally:
            self._tta_n_augments = orig

    def _get_model_info(self) -> dict[str, Any]:
        """Return model architecture and device info."""
        return {
            "architecture": "EfficientNet-B3",
            "num_classes": 5,
            "input_size": 224,
            "device": self.device,
            "iqa_loaded": self.iqa_loaded,
            "glaucoma_loaded": self.glaucoma_loaded,
            "ensemble_size": self.ensemble_size + (1 if self.is_loaded else 0),
            "temperature": round(self.temperature_value, 4),
        }

    def _get_meta(self) -> dict[str, Any]:
        """Return metadata about model versions and paths."""
        model_versions = {}
        if self.dr_checkpoint_path:
            model_versions["dr"] = self.dr_checkpoint_path
        if self.iqa_checkpoint_path:
            model_versions["iqa"] = self.iqa_checkpoint_path
        if self.glaucoma_checkpoint_path:
            model_versions["glaucoma"] = self.glaucoma_checkpoint_path
        for i, path in enumerate(self.ensemble_checkpoint_paths):
            model_versions[f"dr_ensemble_{i}"] = path

        return {
            "model_versions": model_versions,
            "ensemble_size": self.ensemble_size + (1 if self.is_loaded else 0),
            "temperature_scaling": round(self.temperature_value, 4),
        }

    def get_models_info(self) -> dict[str, Any]:
        """
        Return detailed information about all loaded models.
        Used by the /v1/models/info endpoint.
        """
        models = []

        if self.is_loaded and self.model is not None:
            param_count = sum(p.numel() for p in self.model.parameters())
            models.append({
                "name": "dr_efficientnet_b3",
                "type": "classification",
                "architecture": "EfficientNet-B3",
                "checkpoint": self.dr_checkpoint_path,
                "parameters": param_count,
                "parameters_human": f"{param_count / 1e6:.1f}M",
                "device": self.device,
                "status": "loaded",
            })

        if self.iqa_loaded and self.iqa_model is not None:
            param_count = sum(p.numel() for p in self.iqa_model.parameters())
            models.append({
                "name": "iqa_mobilenetv3",
                "type": "quality_assessment",
                "architecture": "MobileNetV3-Small",
                "checkpoint": self.iqa_checkpoint_path,
                "parameters": param_count,
                "parameters_human": f"{param_count / 1e6:.1f}M",
                "device": self.device,
                "status": "loaded",
            })

        if self.glaucoma_loaded and self.glaucoma_model is not None:
            param_count = sum(p.numel() for p in self.glaucoma_model.parameters())
            models.append({
                "name": "glaucoma_unet",
                "type": "segmentation",
                "architecture": "U-Net (EfficientNet-B3 encoder)",
                "checkpoint": self.glaucoma_checkpoint_path,
                "parameters": param_count,
                "parameters_human": f"{param_count / 1e6:.1f}M",
                "device": self.device,
                "status": "loaded",
            })

        for i, (ens_model, ens_path) in enumerate(
            zip(self.ensemble_models, self.ensemble_checkpoint_paths)
        ):
            param_count = sum(p.numel() for p in ens_model.parameters())
            models.append({
                "name": f"dr_ensemble_{i}",
                "type": "classification",
                "architecture": "EfficientNet-B3",
                "checkpoint": ens_path,
                "parameters": param_count,
                "parameters_human": f"{param_count / 1e6:.1f}M",
                "device": self.device,
                "status": "loaded",
            })

        return {
            "models": models,
            "total_loaded": len(models),
            "device": self.device,
            "ensemble_size": self.ensemble_size + (1 if self.is_loaded else 0),
            "temperature_scaling": round(self.temperature_value, 4),
            "inference_metrics": dict(inference_metrics),
        }

    async def check_quality(self, image_bytes: bytes) -> dict[str, Any]:
        """
        Basic image quality assessment using image statistics.
        (Placeholder until a dedicated IQA model is trained.)

        Checks:
        - Brightness (mean pixel value)
        - Contrast (std of pixel values)
        - Resolution (minimum dimension)
        - Sharpness (Laplacian variance)

        Returns:
            dict with score (0-1), passed (bool), and per-check details

        Raises:
            ValueError: If image fails basic validation.
        """
        _validate_image_input(image_bytes)

        def _check():
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return {
                    "score": 0.0,
                    "passed": False,
                    "details": {"error": "Could not decode image"},
                }

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Brightness check
            brightness = float(np.mean(img_rgb)) / 255.0
            brightness_ok = 0.15 < brightness < 0.85
            brightness_score = 1.0 if brightness_ok else max(0, 1.0 - abs(brightness - 0.5) * 2)

            # Contrast check
            contrast = float(np.std(img_rgb))
            contrast_ok = contrast > 25.0
            contrast_score = min(1.0, contrast / 60.0)

            # Resolution check
            min_dim = min(h, w)
            resolution_ok = min_dim >= 256
            resolution_score = min(1.0, min_dim / 512.0)

            # Sharpness (Laplacian variance)
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            sharpness_ok = laplacian_var > 50.0
            sharpness_score = min(1.0, laplacian_var / 200.0)

            # Weighted overall score
            overall = (
                0.25 * brightness_score
                + 0.25 * contrast_score
                + 0.20 * resolution_score
                + 0.30 * sharpness_score
            )

            return {
                "score": round(overall, 3),
                "passed": overall >= 0.45,
                "details": {
                    "brightness": round(brightness_score, 3),
                    "contrast": round(contrast_score, 3),
                    "resolution": round(resolution_score, 3),
                    "sharpness": round(sharpness_score, 3),
                    "image_size": [w, h],
                    "brightness_raw": round(brightness, 3),
                    "contrast_raw": round(contrast, 1),
                    "laplacian_variance": round(laplacian_var, 1),
                },
            }

        try:
            inference_metrics["quality_checks"] += 1
            return await asyncio.to_thread(_check)
        except Exception as e:
            logger.error("Quality check failed: %s", e)
            return {"score": 0.0, "passed": False, "details": {"error": str(e)}}


# ── Async wrappers with caching ───────────────────────────────────────────

async def analyze_fundus_async(
    image_bytes: bytes,
    service: InferenceService | None = None,
    use_tta: Optional[bool] = None,
    high_confidence_mode: bool = False,
) -> dict[str, Any]:
    """
    Async-first DR analysis with Redis caching.

    1. Computes SHA-256 hash of image_bytes
    2. Checks Redis cache first
    3. If cache miss, runs inference in thread pool via InferenceService
    4. Caches the result (minus gradcam to save memory)
    5. Returns result
    """
    from server.services.cache import cache

    if service is None:
        raise RuntimeError("InferenceService must be provided.")

    # Compute image hash
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    # Include TTA/mode in cache key to differentiate results
    cache_suffix = ""
    if high_confidence_mode:
        cache_suffix = ":hc"
    elif use_tta:
        cache_suffix = ":tta"
    cache_key = f"{image_hash}{cache_suffix}"

    # Check cache
    cached = await cache.get_prediction_cache(cache_key)
    if cached is not None:
        inference_metrics["total_cache_hits"] += 1
        logger.info("Cache HIT for image hash %s...", image_hash[:12])
        return cached

    inference_metrics["total_cache_misses"] += 1

    # Run inference (already uses thread pool internally)
    result = await service.analyze_fundus(
        image_bytes,
        use_tta=use_tta,
        high_confidence_mode=high_confidence_mode,
    )

    # Cache the result without gradcam (to save memory)
    cache_result = {k: v for k, v in result.items() if k != "gradcam"}
    await cache.set_prediction_cache(cache_key, cache_result, ttl=3600)

    return result


async def check_quality_async(
    image_bytes: bytes,
    service: InferenceService | None = None,
) -> dict[str, Any]:
    """
    Async quality check with Redis caching.

    Caches quality results by image hash so repeated uploads of the
    same image skip reprocessing.
    """
    from server.services.cache import cache

    if service is None:
        raise RuntimeError("InferenceService must be provided.")

    image_hash = hashlib.sha256(image_bytes).hexdigest()

    # Check cache (use a quality-specific prefix)
    cached = await cache.get_prediction_cache(f"quality:{image_hash}")
    if cached is not None:
        logger.info("Quality cache HIT for image hash %s...", image_hash[:12])
        return cached

    # Run quality check (already uses thread pool internally)
    result = await service.check_quality(image_bytes)

    # Cache it
    await cache.set_prediction_cache(f"quality:{image_hash}", result, ttl=3600)

    return result
