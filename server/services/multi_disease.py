"""
Multi-label eye disease classifier service.

Uses an EfficientNet-B0 trained on ODIR-5K to detect 8 conditions:
Normal(N), Diabetic Retinopathy(D), Glaucoma(G), Cataract(C),
Age-related Macular Degeneration(A), Hypertensive Retinopathy(H),
Pathological Myopia(M), Other Abnormality(O).
"""

import io
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import timm

logger = logging.getLogger(__name__)

# ── Condition metadata ───────────────────────────────────────────────────────
LABEL_CODES = ["N", "D", "G", "C", "A", "H", "M", "O"]

CONDITION_INFO = {
    "N": {
        "name": "Normal",
        "description": "No significant ocular pathology detected in the fundus image.",
        "why_it_matters": "A normal result is reassuring but regular screening is still recommended.",
        "what_to_do": "Continue routine eye examinations as recommended by your eye care provider.",
        "critical": False,
        "threshold": 0.5,
        "risk_level": "none",
    },
    "D": {
        "name": "Diabetic Retinopathy",
        "description": "Damage to retinal blood vessels caused by diabetes, leading to microaneurysms, hemorrhages, and potential vision loss.",
        "why_it_matters": "Leading cause of blindness in working-age adults; early treatment can prevent 95% of severe vision loss.",
        "what_to_do": "See an ophthalmologist for a dilated eye exam, optimize blood sugar control, and follow up per clinical guidelines.",
        "critical": True,
        "threshold": 0.5,
        "risk_level": "high",
    },
    "G": {
        "name": "Glaucoma",
        "description": "Damage to the optic nerve, often from elevated intraocular pressure, causing characteristic cupping of the optic disc.",
        "why_it_matters": "Can cause irreversible peripheral vision loss progressing to blindness if untreated.",
        "what_to_do": "See an ophthalmologist for intraocular pressure measurement, visual field testing, and OCT of the optic nerve.",
        "critical": True,
        "threshold": 0.3,
        "risk_level": "high",
    },
    "C": {
        "name": "Cataract",
        "description": "Clouding of the eye's natural lens, visible as lens opacity in the fundus image.",
        "why_it_matters": "Most common cause of reversible blindness worldwide; significantly impacts quality of life.",
        "what_to_do": "Consult an ophthalmologist for a slit-lamp examination to assess cataract severity and discuss surgical options if vision is impaired.",
        "critical": False,
        "threshold": 0.5,
        "risk_level": "moderate",
    },
    "A": {
        "name": "Age-related Macular Degeneration",
        "description": "Degeneration of the macula causing drusen, pigment changes, and potential central vision loss.",
        "why_it_matters": "Leading cause of irreversible central vision loss in people over 50; early detection enables treatment that can slow progression.",
        "what_to_do": "See a retina specialist for OCT imaging and fluorescein angiography; consider AREDS2 supplements and monitor with an Amsler grid.",
        "critical": True,
        "threshold": 0.3,
        "risk_level": "high",
    },
    "H": {
        "name": "Hypertensive Retinopathy",
        "description": "Retinal vascular changes caused by systemic hypertension, including arteriolar narrowing, hemorrhages, and cotton-wool spots.",
        "why_it_matters": "Indicates end-organ damage from hypertension and is associated with increased cardiovascular risk.",
        "what_to_do": "Consult your physician for blood pressure management and an ophthalmologist for retinal follow-up.",
        "critical": True,
        "threshold": 0.5,
        "risk_level": "moderate",
    },
    "M": {
        "name": "Pathological Myopia",
        "description": "Severe near-sightedness with degenerative changes to the retina including thinning, lacquer cracks, and possible macular atrophy.",
        "why_it_matters": "Can lead to retinal detachment, macular degeneration, and irreversible vision loss.",
        "what_to_do": "See a retina specialist for regular monitoring with OCT and dilated fundus examination; watch for sudden flashes or floaters.",
        "critical": False,
        "threshold": 0.5,
        "risk_level": "moderate",
    },
    "O": {
        "name": "Other Abnormality",
        "description": "Other ocular findings not classified in the primary categories, which may include epiretinal membrane, macular hole, or vascular occlusion.",
        "why_it_matters": "May represent a treatable condition that requires further clinical evaluation.",
        "what_to_do": "Consult an ophthalmologist for a comprehensive examination to identify the specific condition and determine treatment options.",
        "critical": False,
        "threshold": 0.5,
        "risk_level": "low",
    },
}

# ── Preprocessing ────────────────────────────────────────────────────────────
_inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class MultiDiseaseClassifier:
    """Multi-label eye disease classifier using EfficientNet-B0 trained on ODIR-5K."""

    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.device: str = "cpu"
        self.is_loaded: bool = False

    def load_model(self, checkpoint_path: str, device: str = "mps") -> None:
        """Load the trained EfficientNet-B0 multi-label model.

        Args:
            checkpoint_path: Path to the saved checkpoint (.pth).
            device: Device to run inference on.
        """
        self.device = device
        path = Path(checkpoint_path)

        if not path.exists():
            logger.warning(
                "Multi-disease checkpoint not found at %s. "
                "Service will start without multi-disease classifier.",
                checkpoint_path,
            )
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

            num_classes = checkpoint.get("num_classes", 8)
            model_name = checkpoint.get("model_name", "efficientnet_b0")

            self.model = timm.create_model(
                f"{model_name}.ra_in1k", pretrained=False, num_classes=num_classes
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(device)
            self.model.eval()
            self.is_loaded = True

            val_auc = checkpoint.get("val_auc", "N/A")
            epoch = checkpoint.get("epoch", "N/A")
            logger.info(
                "Multi-disease model loaded on %s (epoch %s, val_AUC=%s)",
                device, epoch, val_auc,
            )
        except Exception as e:
            logger.error("Failed to load multi-disease model: %s", e, exc_info=True)
            self.is_loaded = False

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        """Run multi-label classification on a fundus image.

        Args:
            image_bytes: Raw image bytes (JPEG/PNG).

        Returns:
            Dictionary with conditions, detected_conditions, critical_flags,
            overall_risk, and summary.
        """
        if not self.is_loaded or self.model is None:
            return {
                "conditions": [],
                "detected_conditions": [],
                "critical_flags": [],
                "overall_risk": "unknown",
                "summary": "Multi-disease classifier is not available.",
                "error": "Model not loaded",
            }

        # Preprocess
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = _inference_transform(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # Build condition results
        conditions = []
        detected_conditions = []
        critical_flags = []
        worst_risk = "none"
        risk_order = {"none": 0, "low": 1, "moderate": 2, "high": 3}

        for i, code in enumerate(LABEL_CODES):
            info = CONDITION_INFO[code]
            probability = float(round(probs[i], 4))
            threshold = info["threshold"]
            detected = probability >= threshold

            condition_entry = {
                "name": info["name"],
                "code": code,
                "probability": probability,
                "detected": detected,
                "threshold": threshold,
                "description": info["description"],
                "why_it_matters": info["why_it_matters"],
                "what_to_do": info["what_to_do"],
            }
            conditions.append(condition_entry)

            if detected and code != "N":
                detected_conditions.append(info["name"])
                if info["critical"]:
                    critical_flags.append({
                        "condition": info["name"],
                        "code": code,
                        "probability": probability,
                        "urgency": "Refer to specialist promptly",
                        "what_to_do": info["what_to_do"],
                    })
                # Track worst risk
                if risk_order.get(info["risk_level"], 0) > risk_order.get(worst_risk, 0):
                    worst_risk = info["risk_level"]

        # If nothing detected (or only Normal), risk is none/low
        if not detected_conditions:
            worst_risk = "low" if probs[0] < 0.5 else "none"

        # Build summary
        summary = self._build_summary(detected_conditions, critical_flags, worst_risk)

        return {
            "conditions": conditions,
            "detected_conditions": detected_conditions,
            "critical_flags": critical_flags,
            "overall_risk": worst_risk,
            "summary": summary,
        }

    def _build_summary(
        self,
        detected: list[str],
        critical: list[dict],
        risk: str,
    ) -> str:
        """Generate a human-readable summary of the screening results."""
        if not detected:
            return (
                "AI screening did not detect signs of any specific eye disease. "
                "The fundus image appears within normal limits. "
                "Continue routine eye examinations as recommended."
            )

        parts = ["AI screening detected signs of: " + ", ".join(detected) + "."]

        if critical:
            crit_names = [c["condition"] for c in critical]
            parts.append(
                f"IMPORTANT: {', '.join(crit_names)} require(s) prompt specialist evaluation."
            )

        if risk == "high":
            parts.append("Overall risk is assessed as HIGH. Timely ophthalmological referral is recommended.")
        elif risk == "moderate":
            parts.append("Overall risk is assessed as MODERATE. Follow-up with an eye care provider is recommended.")

        parts.append("This is an AI screening result and does not replace a clinical examination.")

        return " ".join(parts)
