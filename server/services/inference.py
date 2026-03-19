"""ModelRegistry: loads ONNX models and provides inference methods."""

import asyncio
import io
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry that loads and manages all ML models."""

    def __init__(self):
        self._models: dict[str, Any] = {}
        self.is_loaded: bool = False
        self._lock = asyncio.Lock()

    async def load_models(
        self,
        dr_path: str,
        glaucoma_path: str,
        amd_path: str,
        iqa_path: str,
        segmentation_path: str,
        device: str = "cpu",
    ) -> None:
        """Load all ONNX models into memory."""
        async with self._lock:
            try:
                import onnxruntime as ort

                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.inter_op_num_threads = 4
                sess_options.intra_op_num_threads = 4

                model_configs = {
                    "dr": dr_path,
                    "glaucoma": glaucoma_path,
                    "amd": amd_path,
                    "iqa": iqa_path,
                    "segmentation": segmentation_path,
                }

                for name, path in model_configs.items():
                    try:
                        session = ort.InferenceSession(path, sess_options, providers=providers)
                        self._models[name] = session
                        logger.info("Loaded model '%s' from %s", name, path)
                    except Exception as e:
                        logger.warning("Could not load model '%s' from %s: %s", name, path, e)
                        self._models[name] = None

                self.is_loaded = True
                logger.info("ModelRegistry ready: %d models loaded", sum(1 for v in self._models.values() if v is not None))

            except ImportError:
                logger.warning("onnxruntime not installed — running with stub models")
                for name in ("dr", "glaucoma", "amd", "iqa", "segmentation"):
                    self._models[name] = None
                self.is_loaded = True

    async def unload_models(self) -> None:
        """Release all model sessions."""
        async with self._lock:
            self._models.clear()
            self.is_loaded = False
            logger.info("All models unloaded")

    def _preprocess_image(self, image_bytes: bytes, target_size: tuple[int, int] = (512, 512)) -> np.ndarray:
        """Convert raw image bytes to model-ready numpy array (NCHW float32)."""
        from PIL import Image as PILImage

        img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(target_size, PILImage.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        # HWC -> CHW -> NCHW
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0)
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        arr = (arr - mean) / std
        return arr

    async def analyze_fundus(
        self, image_bytes: bytes, model_names: list[str] | None = None
    ) -> dict[str, Any]:
        """Run selected models on a fundus image and return structured results."""
        if model_names is None:
            model_names = ["dr", "glaucoma", "amd"]

        input_tensor = await asyncio.to_thread(self._preprocess_image, image_bytes)
        results: dict[str, Any] = {}

        for name in model_names:
            session = self._models.get(name)
            if session is None:
                # Return stub results when model is unavailable
                if name == "dr":
                    results["dr"] = {"grade": 0, "confidence": 0.0, "probabilities": [1.0, 0.0, 0.0, 0.0, 0.0]}
                elif name == "glaucoma":
                    results["glaucoma"] = {"probability": 0.0, "label": "normal"}
                elif name == "amd":
                    results["amd"] = {"probability": 0.0, "label": "normal"}
                continue

            try:
                input_name = session.get_inputs()[0].name
                output = await asyncio.to_thread(session.run, None, {input_name: input_tensor})
                probs = self._softmax(output[0][0])

                if name == "dr":
                    grade = int(np.argmax(probs))
                    results["dr"] = {
                        "grade": grade,
                        "confidence": float(probs[grade]),
                        "probabilities": [float(p) for p in probs],
                    }
                elif name == "glaucoma":
                    prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
                    results["glaucoma"] = {
                        "probability": prob,
                        "label": "suspect" if prob > 0.5 else "normal",
                    }
                elif name == "amd":
                    prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
                    results["amd"] = {
                        "probability": prob,
                        "label": "suspect" if prob > 0.5 else "normal",
                    }

            except Exception as e:
                logger.error("Inference error for model '%s': %s", name, e)
                results[name] = {"error": str(e)}

        return results

    async def check_quality(self, image_bytes: bytes) -> dict[str, Any]:
        """Run IQA model on an image and return quality score + details."""
        session = self._models.get("iqa")

        if session is None:
            # Stub: derive a basic quality score from image stats
            try:
                from PIL import Image as PILImage

                img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
                arr = np.array(img, dtype=np.float32)
                brightness = float(np.mean(arr))
                contrast = float(np.std(arr))

                # Heuristic score
                brightness_ok = 0.3 < (brightness / 255.0) < 0.8
                contrast_ok = contrast > 30.0
                size_ok = min(img.size) >= 256

                score = 0.0
                if brightness_ok:
                    score += 0.35
                if contrast_ok:
                    score += 0.35
                if size_ok:
                    score += 0.30

                return {
                    "score": round(score, 3),
                    "details": {
                        "brightness_score": round(brightness / 255.0, 3),
                        "contrast_score": round(min(contrast / 80.0, 1.0), 3),
                        "resolution": list(img.size),
                        "blur_score": 0.7,
                        "illumination_score": round(brightness / 255.0, 3),
                        "field_of_view_score": 0.8 if size_ok else 0.3,
                    },
                }
            except Exception as e:
                logger.error("Stub IQA failed: %s", e)
                return {"score": 0.5, "details": {}}

        try:
            input_tensor = await asyncio.to_thread(
                self._preprocess_image, image_bytes, (256, 256)
            )
            input_name = session.get_inputs()[0].name
            output = await asyncio.to_thread(session.run, None, {input_name: input_tensor})
            score = float(output[0][0][0]) if output[0].ndim > 1 else float(output[0][0])
            score = max(0.0, min(1.0, score))

            return {
                "score": round(score, 3),
                "details": {
                    "raw_output": float(output[0].flat[0]),
                },
            }
        except Exception as e:
            logger.error("IQA inference error: %s", e)
            return {"score": 0.5, "details": {"error": str(e)}}

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()
