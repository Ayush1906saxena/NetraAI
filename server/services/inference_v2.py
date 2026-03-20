"""
Simplified inference service using the trained EfficientNet-B3 DR model.

Loads the actual trained checkpoint from checkpoints/dr_aptos/best.pth
and provides analyze_fundus() and check_quality() methods.

Enhanced with:
- Async wrappers that run inference in a thread pool
- Redis caching for prediction results (keyed by image SHA-256)
- Inference request counting and latency metrics
- Model warm-up on load (dummy inference to pre-compile)
"""

import asyncio
import base64
import hashlib
import io
import logging
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

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


class InferenceService:
    """
    Loads the trained EfficientNet-B3 checkpoint and runs DR inference
    with preprocessing, GradCAM, and quality checks.
    """

    def __init__(self):
        self.model: Optional[DRGrader] = None
        self.device: str = "cpu"
        self.is_loaded: bool = False
        self._preprocessor = None
        self._lock = asyncio.Lock()

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
            logger.info("DR model loaded successfully. Ready for inference.")

            # Warm-up: run a dummy inference to pre-compile model graph
            await self._warmup()

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
        """Release model from memory."""
        async with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
                self.is_loaded = False
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                logger.info("DR model unloaded.")

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

    async def analyze_fundus(self, image_bytes: bytes) -> dict[str, Any]:
        """
        Full DR analysis pipeline:
        1. Validate input image
        2. Preprocess (circle crop, Ben Graham, CLAHE, normalize)
        3. Run EfficientNet-B3 inference
        4. Generate GradCAM heatmap
        5. Compute referral recommendation

        Args:
            image_bytes: Raw image file bytes (JPEG/PNG)

        Returns:
            dict with grade, probabilities, confidence, referral info, gradcam base64

        Raises:
            ValueError: If image fails validation (bad format, size).
            RuntimeError: If model is not loaded.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

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

        # ── Inference ─────────────────────────────────────────────────
        def _infer():
            t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    input_t = input_tensor.to(self.device)
                    logits = self.model(input_t)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            except Exception as exc:
                logger.error("Model inference failed: %s", exc, exc_info=True)
                raise RuntimeError(
                    "Model inference failed. The image may be incompatible or "
                    "the model encountered a numerical error."
                ) from exc
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "DR inference completed in %.1fms on %s", latency_ms, self.device
            )
            return probs, latency_ms

        probs, latency_ms = await asyncio.to_thread(_infer)

        # Update metrics
        inference_metrics["total_requests"] += 1
        inference_metrics["total_latency_ms"] += latency_ms
        inference_metrics["avg_latency_ms"] = (
            inference_metrics["total_latency_ms"] / inference_metrics["total_requests"]
        )

        grade = int(np.argmax(probs))
        confidence = float(probs[grade])

        # GradCAM
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

        # Referral logic
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

        return {
            "dr": {
                "grade": grade,
                "grade_name": DR_GRADE_NAMES[grade],
                "description": DR_GRADE_DESCRIPTIONS[grade],
                "confidence": round(confidence, 4),
                "probabilities": {
                    DR_GRADE_NAMES[i]: round(float(probs[i]), 4) for i in range(5)
                },
                "probabilities_list": [round(float(p), 4) for p in probs],
            },
            "referral": {
                "is_referable": is_referable,
                "urgency": urgency,
                "risk_level": risk,
                "recommendation": (
                    f"Refer to ophthalmologist ({urgency}): {DR_GRADE_NAMES[grade]} detected."
                    if is_referable
                    else "No immediate referral needed. Routine follow-up in 12 months."
                ),
            },
            "gradcam": {
                "overlay_png_base64": gradcam_b64,
                "has_gradcam": bool(gradcam_b64),
            },
            "model_info": {
                "architecture": "EfficientNet-B3",
                "num_classes": 5,
                "input_size": 224,
                "device": self.device,
            },
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

    # Check cache
    cached = await cache.get_prediction_cache(image_hash)
    if cached is not None:
        inference_metrics["total_cache_hits"] += 1
        logger.info("Cache HIT for image hash %s...", image_hash[:12])
        return cached

    inference_metrics["total_cache_misses"] += 1

    # Run inference (already uses thread pool internally)
    loop = asyncio.get_event_loop()
    result = await service.analyze_fundus(image_bytes)

    # Cache the result without gradcam (to save memory)
    cache_result = {k: v for k, v in result.items() if k != "gradcam"}
    await cache.set_prediction_cache(image_hash, cache_result, ttl=3600)

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
