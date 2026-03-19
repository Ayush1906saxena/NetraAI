"""
GradCAM visualization for DR grading models.

Generates class activation maps highlighting regions the model considers
most important for its prediction. Essential for clinical explainability:
ophthalmologists need to verify the model is looking at real pathology
(microaneurysms, hemorrhages, exudates) and not artifacts.
"""

import io
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Works with ViT-based models (RETFound) by hooking into the last
    attention block's output and using gradient information to produce
    spatial heatmaps.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Args:
            model: The DR grading model (RETFoundDRGrader or similar).
            target_layer: The layer to hook for activations.
                          If None, attempts to auto-detect the last
                          transformer block or conv layer.
        """
        self.model = model
        self.model.eval()

        self.target_layer = target_layer or self._find_target_layer()
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._forward_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _find_target_layer(self) -> nn.Module:
        """
        Auto-detect the best target layer for GradCAM.

        For ViT models: last transformer block's LayerNorm or MLP.
        For CNN models: last convolutional layer.
        """
        # Try ViT-style: backbone.blocks[-1].norm1 or backbone.blocks[-1]
        for attr_path in [
            "backbone.base_model.model.blocks",  # LoRA-wrapped
            "backbone.blocks",  # direct ViT
        ]:
            obj = self.model
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                # obj should be a ModuleList of transformer blocks
                if hasattr(obj, "__getitem__"):
                    return obj[-1]
            except AttributeError:
                continue

        # Try CNN-style: last layer with Conv2d
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            return last_conv

        raise ValueError(
            "Could not auto-detect target layer. "
            "Please pass target_layer explicitly."
        )

    def _save_activation(self, module, input, output):
        """Forward hook to capture activations."""
        if isinstance(output, tuple):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook to capture gradients."""
        if isinstance(grad_output, tuple):
            self.gradients = grad_output[0].detach()
        else:
            self.gradients = grad_output.detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap.

        Args:
            input_tensor: (1, 3, H, W) preprocessed input tensor.
            target_class: Class index to explain. If None, uses predicted class.

        Returns:
            (H, W) heatmap normalized to [0, 1].
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.requires_grad_(True)
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients and backward pass for target class
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not capture activations/gradients.")

        gradients = self.gradients
        activations = self.activations

        # For ViT: activations shape is (B, num_tokens, dim)
        # For CNN: activations shape is (B, C, H, W)
        if activations.dim() == 3:
            # ViT: reshape tokens to spatial grid
            return self._gradcam_vit(activations, gradients, input_tensor.shape[-2:])
        else:
            # CNN: standard GradCAM
            return self._gradcam_cnn(activations, gradients, input_tensor.shape[-2:])

    def _gradcam_vit(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
        spatial_size: tuple,
    ) -> np.ndarray:
        """GradCAM for Vision Transformer outputs."""
        # activations: (B, num_tokens, dim), gradients: same shape
        # Remove CLS token if present (first token)
        act = activations[0, 1:, :]  # (num_patches, dim)
        grad = gradients[0, 1:, :]

        # Weights: global average of gradients per feature dimension
        weights = grad.mean(dim=0)  # (dim,)

        # Weighted sum of activations
        cam = (act * weights).sum(dim=-1)  # (num_patches,)
        cam = F.relu(cam)

        # Reshape to spatial grid
        num_patches = cam.shape[0]
        grid_size = int(num_patches**0.5)
        if grid_size * grid_size != num_patches:
            grid_size = int(np.ceil(num_patches**0.5))
            padded = torch.zeros(grid_size * grid_size, device=cam.device)
            padded[:num_patches] = cam
            cam = padded

        cam = cam.view(grid_size, grid_size)

        # Upsample to input spatial resolution
        cam = cam.unsqueeze(0).unsqueeze(0).float()
        cam = F.interpolate(cam, size=spatial_size, mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def _gradcam_cnn(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
        spatial_size: tuple,
    ) -> np.ndarray:
        """Standard GradCAM for CNN feature maps."""
        # activations: (B, C, H, W), gradients: same
        weights = gradients[0].mean(dim=(1, 2))  # (C,)
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * activations[0]).sum(dim=0)
        cam = F.relu(cam)

        # Upsample
        cam = cam.unsqueeze(0).unsqueeze(0).float()
        cam = F.interpolate(cam, size=spatial_size, mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        """Remove registered hooks to free memory."""
        self._forward_hook.remove()
        self._backward_hook.remove()

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass


def generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    target_class: Optional[int] = None,
    target_layer: Optional[nn.Module] = None,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.5,
) -> bytes:
    """
    Generate a GradCAM heatmap overlay and return it as PNG bytes.

    This is the main entry point for GradCAM visualization.

    Args:
        model: DR grading model.
        input_tensor: (1, 3, H, W) or (3, H, W) preprocessed tensor.
        original_image: (H, W, 3) uint8 RGB image for overlay.
        target_class: Class to explain (None = predicted class).
        target_layer: Layer to hook (None = auto-detect).
        colormap: OpenCV colormap for heatmap.
        alpha: Overlay transparency (0 = full image, 1 = full heatmap).

    Returns:
        PNG image bytes of the heatmap overlay.
    """
    gradcam = GradCAM(model, target_layer=target_layer)

    try:
        heatmap = gradcam.generate(input_tensor, target_class=target_class)
    finally:
        gradcam.remove_hooks()

    # Resize heatmap to match original image dimensions
    h, w = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert heatmap to colormap
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay on original image
    overlay = (
        (1.0 - alpha) * original_image.astype(np.float32)
        + alpha * heatmap_colored.astype(np.float32)
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Encode as PNG bytes
    img_pil = Image.fromarray(overlay)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return buf.getvalue()


def generate_gradcam_grid(
    model: nn.Module,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    num_classes: int = 5,
    target_layer: Optional[nn.Module] = None,
    class_names: Optional[list[str]] = None,
) -> bytes:
    """
    Generate a grid of GradCAM heatmaps for all classes.

    Useful for error analysis: shows what regions the model associates
    with each DR grade for a given image.

    Returns:
        PNG bytes of the grid image.
    """
    if class_names is None:
        class_names = ["No DR", "Mild", "Moderate", "Severe", "PDR"][:num_classes]

    gradcam = GradCAM(model, target_layer=target_layer)

    h, w = original_image.shape[:2]
    cell_size = 256
    padding = 4
    cols = min(num_classes, 5)
    rows = (num_classes + cols - 1) // cols

    grid_w = cols * (cell_size + padding) + padding
    grid_h = rows * (cell_size + padding + 30) + padding  # 30px for text
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    try:
        for idx in range(num_classes):
            heatmap = gradcam.generate(input_tensor.clone(), target_class=idx)
            heatmap_resized = cv2.resize(heatmap, (w, h))
            heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            overlay = (0.5 * original_image.astype(np.float32) + 0.5 * heatmap_colored.astype(np.float32))
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            overlay = cv2.resize(overlay, (cell_size, cell_size))

            row = idx // cols
            col = idx % cols
            y = padding + row * (cell_size + padding + 30)
            x = padding + col * (cell_size + padding)

            # Add class name label
            cv2.putText(
                grid,
                class_names[idx],
                (x, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            grid[y + 30: y + 30 + cell_size, x: x + cell_size] = overlay
    finally:
        gradcam.remove_hooks()

    img_pil = Image.fromarray(grid)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return buf.getvalue()
