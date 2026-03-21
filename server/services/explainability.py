"""
Lesion-level explainability from GradCAM heatmaps.

Analyzes spatial GradCAM activation maps to produce clinically meaningful
descriptions of where the AI model focused and what findings those regions
may correspond to based on the predicted DR grade.

Approach:
1. Divide the fundus image into anatomical regions (quadrants + macular center + peripapillary)
2. Compute mean GradCAM activation per region
3. Regions above threshold = findings
4. Map (grade + region) -> clinical description using a DR pathology lookup table
"""

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Region definitions ────────────────────────────────────────────────────
# Regions are defined as relative coordinates within a square fundus image.
# Each region: (name, center_y_frac, center_x_frac, radius_frac)
# The macular region is central; peripapillary is nasal-side of center.

REGION_DEFINITIONS: list[dict[str, Any]] = [
    {"name": "macular", "cy": 0.50, "cx": 0.50, "r": 0.15, "priority": 1},
    {"name": "peripapillary", "cy": 0.50, "cx": 0.30, "r": 0.12, "priority": 2},
    {"name": "temporal", "cy": 0.50, "cx": 0.75, "r": 0.18, "priority": 3},
    {"name": "nasal", "cy": 0.50, "cx": 0.25, "r": 0.18, "priority": 4},
    {"name": "superior", "cy": 0.25, "cx": 0.50, "r": 0.20, "priority": 5},
    {"name": "inferior", "cy": 0.75, "cx": 0.50, "r": 0.20, "priority": 6},
]

# ── Activation intensity classification ───────────────────────────────────
INTENSITY_THRESHOLDS = {
    "severe": 0.7,
    "moderate": 0.4,
    "mild": 0.2,
}

# ── Clinical finding lookup: grade -> region -> description ───────────────
# Maps (DR grade, region) to likely clinical findings based on known
# pathology patterns in diabetic retinopathy.

FINDING_LOOKUP: dict[int, dict[str, str]] = {
    0: {
        "macular": "No significant findings in the macular region",
        "peripapillary": "No significant findings near the optic disc",
        "temporal": "No significant findings in the temporal region",
        "nasal": "No significant findings in the nasal region",
        "superior": "No significant findings in the superior region",
        "inferior": "No significant findings in the inferior region",
    },
    1: {
        "macular": "Likely microaneurysms near the macula",
        "peripapillary": "Possible microaneurysms near the optic disc",
        "temporal": "Scattered microaneurysms in the temporal arcade",
        "nasal": "Microaneurysms in the nasal retina",
        "superior": "Microaneurysms in the superior arcade",
        "inferior": "Microaneurysms in the inferior arcade",
    },
    2: {
        "macular": "Hemorrhages and hard exudates threatening the macula",
        "peripapillary": "Hemorrhages and possible cotton-wool spots near the disc",
        "temporal": "Dot-blot hemorrhages and hard exudates in temporal retina",
        "nasal": "Hemorrhages and exudates in the nasal quadrant",
        "superior": "Hemorrhages and venous beading in the superior arcade",
        "inferior": "Hemorrhages and possible IRMA in the inferior arcade",
    },
    3: {
        "macular": "Extensive hemorrhages and macular edema risk",
        "peripapillary": "Cotton-wool spots and venous beading near the disc",
        "temporal": "Extensive hemorrhages with intraretinal microvascular abnormalities (IRMA)",
        "nasal": "Venous beading and extensive hemorrhages in nasal quadrant",
        "superior": "Severe hemorrhages with venous abnormalities superiorly",
        "inferior": "IRMA and extensive hemorrhages in the inferior arcade",
    },
    4: {
        "macular": "Neovascularization threatening the macula with high risk of tractional detachment",
        "peripapillary": "Neovascularization of the disc (NVD) — high-risk proliferative finding",
        "temporal": "Neovascularization elsewhere (NVE) in temporal retina with preretinal hemorrhage",
        "nasal": "Neovascularization and fibrovascular proliferation in nasal retina",
        "superior": "Neovascularization with possible vitreous hemorrhage superiorly",
        "inferior": "Neovascularization elsewhere (NVE) with fibrous proliferation inferiorly",
    },
}

# ── Grade-level context ───────────────────────────────────────────────────
GRADE_CONTEXT: dict[int, str] = {
    0: "normal retinal appearance",
    1: "early microaneurysm formation (mild NPDR)",
    2: "moderate vascular damage with hemorrhages and exudates (moderate NPDR)",
    3: "severe vascular compromise with risk of proliferative progression (severe NPDR)",
    4: "proliferative disease with abnormal new vessel growth (PDR)",
}


def _create_region_mask(
    height: int,
    width: int,
    cy_frac: float,
    cx_frac: float,
    r_frac: float,
) -> np.ndarray:
    """Create a circular boolean mask for a region."""
    cy = int(cy_frac * height)
    cx = int(cx_frac * width)
    r = int(r_frac * min(height, width))

    y_coords, x_coords = np.ogrid[:height, :width]
    dist_sq = (y_coords - cy) ** 2 + (x_coords - cx) ** 2
    mask = dist_sq <= r ** 2
    return mask


def _classify_intensity(activation: float) -> Optional[str]:
    """Classify activation level into severity category."""
    for level, threshold in INTENSITY_THRESHOLDS.items():
        if activation >= threshold:
            return level
    return None


def analyze_gradcam(
    heatmap: np.ndarray,
    original_image: np.ndarray,
    dr_grade: int,
    activation_threshold: float = 0.2,
) -> dict[str, Any]:
    """
    Analyze a GradCAM heatmap to produce lesion-level explainability.

    Args:
        heatmap: GradCAM activation map as a 2D numpy array (H, W) with values
                 in [0, 1]. Higher values indicate stronger model attention.
        original_image: The original fundus image as numpy array (H, W, C).
                       Used for spatial reference (dimensions).
        dr_grade: The predicted DR grade (0-4).
        activation_threshold: Minimum mean activation in a region to report
                            it as a finding. Default 0.2.

    Returns:
        Dictionary with:
        - findings: list of detected regions with location, intensity,
                   possible_finding, mean_activation
        - overall_description: human-readable summary
        - region_activations: dict of region_name -> mean_activation
        - grade: the input DR grade for reference
    """
    # Normalize heatmap to [0, 1] if needed
    if heatmap.ndim == 3:
        # If 3-channel heatmap, convert to single channel
        heatmap = np.mean(heatmap, axis=-1)

    hmap_min = heatmap.min()
    hmap_max = heatmap.max()
    if hmap_max > hmap_min:
        heatmap = (heatmap - hmap_min) / (hmap_max - hmap_min)
    else:
        heatmap = np.zeros_like(heatmap)

    h, w = heatmap.shape[:2]

    # Clamp grade to valid range
    grade = max(0, min(4, dr_grade))

    # Compute per-region activations
    region_activations: dict[str, float] = {}
    findings: list[dict[str, Any]] = []

    for region_def in REGION_DEFINITIONS:
        name = region_def["name"]
        mask = _create_region_mask(h, w, region_def["cy"], region_def["cx"], region_def["r"])

        # Compute mean activation within this region
        if mask.sum() > 0:
            mean_activation = float(np.mean(heatmap[mask]))
        else:
            mean_activation = 0.0

        region_activations[name] = round(mean_activation, 4)

        # Only report regions above threshold
        if mean_activation >= activation_threshold:
            intensity = _classify_intensity(mean_activation)
            if intensity is None:
                continue

            possible_finding = FINDING_LOOKUP.get(grade, FINDING_LOOKUP[0]).get(
                name, f"Elevated AI attention in the {name} region"
            )

            findings.append({
                "location": name,
                "intensity": intensity,
                "mean_activation": round(mean_activation, 4),
                "possible_finding": possible_finding,
                "priority": region_def["priority"],
            })

    # Sort findings by activation (highest first), then priority
    findings.sort(key=lambda f: (-f["mean_activation"], f["priority"]))

    # Remove internal priority field from output
    for f in findings:
        f.pop("priority", None)

    # Build overall description
    overall_description = _build_description(findings, grade, region_activations)

    return {
        "findings": findings,
        "overall_description": overall_description,
        "region_activations": region_activations,
        "grade": grade,
        "total_findings": len(findings),
    }


def _build_description(
    findings: list[dict[str, Any]],
    grade: int,
    region_activations: dict[str, float],
) -> str:
    """Build a human-readable overall description."""
    n = len(findings)

    if n == 0:
        if grade == 0:
            return (
                "The AI did not detect any regions of significant concern. "
                "The retinal image appears within normal limits."
            )
        return (
            f"The AI classified this image as {GRADE_CONTEXT.get(grade, 'unknown')} "
            "but did not identify strongly localized regions of concern in the activation map."
        )

    # Find the primary region (highest activation)
    primary = findings[0]
    primary_location = primary["location"]

    # Count severe findings
    severe_count = sum(1 for f in findings if f["intensity"] == "severe")
    moderate_count = sum(1 for f in findings if f["intensity"] == "moderate")

    if n == 1:
        region_phrase = f"in the {primary_location} region"
    else:
        locations = [f["location"] for f in findings[:3]]
        if len(locations) == 2:
            region_phrase = f"in the {locations[0]} and {locations[1]} regions"
        else:
            region_phrase = (
                f"in the {', '.join(locations[:-1])}, and {locations[-1]} regions"
            )

    grade_desc = GRADE_CONTEXT.get(grade, "")

    description = (
        f"The AI detected {n} region{'s' if n > 1 else ''} of concern, "
        f"primarily {region_phrase}, suggesting {grade_desc}."
    )

    if severe_count > 0:
        description += (
            f" {severe_count} region{'s show' if severe_count > 1 else ' shows'} "
            "high-intensity activation, indicating strong model confidence in pathological findings."
        )

    if primary_location == "macular" and grade >= 2:
        description += (
            " Notably, the macular region shows significant activation — "
            "macular involvement may affect central vision and warrants prompt evaluation."
        )

    return description
