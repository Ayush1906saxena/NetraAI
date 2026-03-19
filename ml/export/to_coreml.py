"""
Export PyTorch DR models to CoreML format for iOS/macOS deployment.

CoreML enables:
- On-device IQA on iPad/iPhone before uploading images.
- Offline-capable DR screening in areas with poor connectivity.
- Apple Neural Engine acceleration for fast inference.
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def export_to_coreml(
    model: nn.Module,
    output_path: str,
    input_shape: tuple = (1, 3, 224, 224),
    model_name: str = "NetraDRGrader",
    class_labels: Optional[list[str]] = None,
    convert_to_float16: bool = True,
    device: str = "cpu",
) -> str:
    """
    Export a PyTorch model to CoreML format (.mlpackage).

    Args:
        model: PyTorch model in eval mode.
        output_path: Destination path (should end with .mlpackage).
        input_shape: (batch, channels, height, width).
        model_name: Human-readable model name in CoreML metadata.
        class_labels: Optional list of class labels for classifier models.
        convert_to_float16: If True, quantize weights to float16 for
                            smaller model size and faster Neural Engine inference.
        device: Device for tracing.

    Returns:
        Path to saved CoreML model.
    """
    try:
        import coremltools as ct
    except ImportError:
        raise ImportError(
            "coremltools is required for CoreML export. "
            "Install with: pip install coremltools"
        )

    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    model = model.to(device).eval()
    dummy_input = torch.randn(*input_shape, device=device)

    # Trace the model
    print("Tracing model with TorchScript ...")
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input)

    # ImageNet normalization: CoreML can handle this natively
    # Input is RGB image [0, 255] -> normalize with ImageNet stats
    scale = 1.0 / (255.0 * 0.226)  # approximate single scale
    bias = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]

    image_input = ct.ImageType(
        name="input_image",
        shape=input_shape,
        scale=scale,
        bias=bias,
        color_layout=ct.colorlayout.RGB,
    )

    print("Converting to CoreML ...")
    if class_labels:
        # Classifier model
        classifier_config = ct.ClassifierConfig(class_labels)
        mlmodel = ct.convert(
            traced,
            inputs=[image_input],
            classifier_config=classifier_config,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS16,
        )
    else:
        mlmodel = ct.convert(
            traced,
            inputs=[image_input],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS16,
        )

    # Set metadata
    mlmodel.author = "Netra AI"
    mlmodel.short_description = f"{model_name} - DR screening model"
    mlmodel.version = "1.0"

    # Quantize to float16 for smaller size and Neural Engine optimization
    if convert_to_float16:
        print("Quantizing to float16 ...")
        mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(
            mlmodel, nbits=16
        )

    mlmodel.save(output_path)
    print(f"CoreML model saved to {output_path}")

    # Report model size
    import os

    if os.path.isdir(output_path):
        total_size = sum(
            f.stat().st_size
            for f in Path(output_path).rglob("*")
            if f.is_file()
        )
    else:
        total_size = Path(output_path).stat().st_size
    size_mb = total_size / (1024 * 1024)
    print(f"CoreML model size: {size_mb:.1f} MB")

    return output_path


def export_iqa_to_coreml(
    checkpoint_path: str,
    output_path: str,
    img_size: int = 224,
    device: str = "cpu",
) -> str:
    """
    Export the IQA model to CoreML for on-device image quality assessment.

    The IQA model runs on the capture device (iPad/iPhone) to give
    real-time feedback before the image is uploaded.

    Args:
        checkpoint_path: Path to IQA model checkpoint.
        output_path: Destination .mlpackage path.
        img_size: Input image size (default 224).
        device: Device for tracing.

    Returns:
        Path to saved CoreML model.
    """
    from ml.models.iqa_model import FundusIQA

    model = FundusIQA.from_checkpoint(checkpoint_path, device=device)

    # The IQA model returns a dict; we need a wrapper for CoreML export
    class IQAExportWrapper(nn.Module):
        """Wrapper that outputs a single tensor [quality, gradeable, guidance...]."""

        def __init__(self, iqa_model: FundusIQA):
            super().__init__()
            self.model = iqa_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.model(x)
            quality = out["quality"]            # (B, 1)
            gradeable = torch.sigmoid(out["gradeable"])  # (B, 1)
            guidance = torch.sigmoid(out["guidance"])     # (B, 8)
            return torch.cat([quality, gradeable, guidance], dim=1)

    wrapper = IQAExportWrapper(model)
    wrapper.eval()

    return export_to_coreml(
        wrapper,
        output_path,
        input_shape=(1, 3, img_size, img_size),
        model_name="NetraIQA",
        convert_to_float16=True,
        device=device,
    )


def export_dr_to_coreml(
    checkpoint_path: str,
    output_path: str,
    num_classes: int = 5,
    img_size: int = 224,
    device: str = "cpu",
) -> str:
    """
    Export the DR grading model to CoreML.

    Args:
        checkpoint_path: Path to DR model checkpoint.
        output_path: Destination .mlpackage path.
        num_classes: Number of DR grades.
        img_size: Input image size.
        device: Device for tracing.

    Returns:
        Path to saved CoreML model.
    """
    from ml.evaluation.evaluate import load_model

    model = load_model(checkpoint_path, device=device, num_classes=num_classes)

    class_labels = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]
    class_labels = class_labels[:num_classes]

    return export_to_coreml(
        model,
        output_path,
        input_shape=(1, 3, img_size, img_size),
        model_name="NetraDRGrader",
        class_labels=class_labels,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export models to CoreML")
    parser.add_argument("--model-type", choices=["dr", "iqa"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", required=True, help="Output .mlpackage path")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.model_type == "dr":
        export_dr_to_coreml(args.checkpoint, args.output, img_size=args.img_size, device=args.device)
    else:
        export_iqa_to_coreml(args.checkpoint, args.output, img_size=args.img_size, device=args.device)
