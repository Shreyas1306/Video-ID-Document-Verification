"""
Visual Document Integrity Classifier
====================================
Binary document classification (REAL vs ATTACKED) using transfer learning with
an EfficientNet-B0 baseline.

Features:
  - EfficientNet-B0 backbone with replaced classification head for binary classification.
  - Penultimate feature embedding extraction for temporal consistency analysis.
  - Softmax prediction with calibrated integrity score: P(REAL).
  - Statistical proxy fallback when model weights are not loaded.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

logger = logging.getLogger(__name__)


class DocumentIntegrityClassifier(nn.Module):
    """
    EfficientNet-B0 baseline for document integrity classification (REAL vs ATTACKED).
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.device = self._resolve_device(device)

        # 1. Initialize backbone
        self.model = self._create_backbone(backbone_name, pretrained)

        # 2. Identify in_features and replace classifier head
        if hasattr(self.model, "classifier"):
            # EfficientNet-B0: classifier is nn.Sequential(Dropout, Linear(1280, 1000))
            if isinstance(self.model.classifier, nn.Sequential):
                in_features = self.model.classifier[1].in_features
            elif isinstance(self.model.classifier, nn.Linear):
                in_features = self.model.classifier.in_features
            else:
                in_features = 1280
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_features, num_classes),
            )
        elif hasattr(self.model, "fc"):
            # ResNet / others fallback
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            raise ValueError(f"Unsupported model architecture: {backbone_name}")

        self.in_features = in_features

        # 3. Load checkpoint if provided
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        self.to(self.device)

    def _resolve_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        if device is None or device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _create_backbone(self, backbone_name: str, pretrained: bool) -> nn.Module:
        """Create model backbone with safe fallback if network weights download fails."""
        if backbone_name.lower() == "efficientnet_b0":
            if pretrained:
                try:
                    weights = models.EfficientNet_B0_Weights.DEFAULT
                    return models.efficientnet_b0(weights=weights)
                except Exception as e:
                    logger.warning(
                        f"Could not load official pretrained weights for {backbone_name} ({e}). "
                        f"Initializing without preloaded ImageNet weights."
                    )
                    return models.efficientnet_b0(weights=None)
            else:
                return models.efficientnet_b0(weights=None)
        elif backbone_name.lower() == "resnet50":
            if pretrained:
                try:
                    weights = models.ResNet50_Weights.DEFAULT
                    return models.resnet50(weights=weights)
                except Exception as e:
                    logger.warning(f"Could not load weights for resnet50 ({e}). Initializing without weights.")
                    return models.resnet50(weights=None)
            else:
                return models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits (B, num_classes)."""
        return self.model(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract penultimate feature embedding vector (B, in_features).
        Bypasses the final linear classifier.
        """
        if hasattr(self.model, "features") and hasattr(self.model, "avgpool"):
            # EfficientNet extraction
            x_feat = self.model.features(x)
            x_feat = self.model.avgpool(x_feat)
            x_feat = torch.flatten(x_feat, 1)
            return x_feat
        else:
            raise NotImplementedError("Feature extraction not implemented for this backbone.")

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities: (B, 2) where [:, 0] is REAL, [:, 1] is ATTACKED."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        image_size: Tuple[int, int] = (224, 224),
    ) -> Dict[str, Any]:
        """
        Run inference on a single image and return structured prediction.
        """
        self.eval()

        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            # Convert BGR (cv2) to RGB
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                rgb_img = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            else:
                rgb_img = image_input
            pil_img = Image.fromarray(rgb_img)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        tensor = transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.predict_proba(tensor).cpu().squeeze(0).numpy()

        p_real = float(probs[0])
        p_attacked = float(probs[1])
        predicted_class = "REAL" if p_real >= 0.5 else "ATTACKED"
        predicted_label = 0 if predicted_class == "REAL" else 1

        return {
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "probability_real": round(p_real, 4),
            "probability_attacked": round(p_attacked, 4),
            "integrity_score": round(p_real, 4),  # Higher integrity = higher confidence of being REAL
        }

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load trained weights from checkpoint."""
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_p}")

        checkpoint = torch.load(ckpt_p, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.load_state_dict(checkpoint)
        logger.info(f"Loaded visual integrity model checkpoint from: {ckpt_p}")


def compute_statistical_proxy_score(image: np.ndarray) -> Dict[str, float]:
    """
    Statistical image-level proxy metrics (sharpness, edge density, color uniformity).
    Used as an evidence fallback if deep learning weights are unavailable.
    """
    if image is None or image.size == 0:
        return {"sharpness": 0.0, "edge_density": 0.0, "color_uniformity": 0.0, "proxy_score": 0.0}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # 1. Sharpness: normalized Laplacian variance
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    norm_sharpness = min(1.0, laplacian_var / 500.0)

    # 2. Edge density: Canny edge ratio
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1]))
    norm_edge = min(1.0, edge_density / 0.20)

    # 3. Color uniformity: standard deviation across channels
    if len(image.shape) == 3:
        std_per_channel = [float(np.std(image[:, :, c])) for c in range(3)]
        avg_std = float(np.mean(std_per_channel))
        norm_color = min(1.0, avg_std / 70.0)
    else:
        norm_color = 0.5

    proxy_score = round(0.4 * norm_sharpness + 0.3 * norm_edge + 0.3 * norm_color, 4)

    return {
        "sharpness": round(norm_sharpness, 4),
        "edge_density": round(norm_edge, 4),
        "color_uniformity": round(norm_color, 4),
        "proxy_score": proxy_score,
    }
