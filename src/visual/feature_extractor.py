"""
Visual Feature Extractor
========================
Extract deep visual feature embeddings from normalized identity document frames
using the trained or pretrained EfficientNet backbone.

Embeddings are L2-normalized vectors (1280-dim for EfficientNet-B0) suitable for
cross-frame cosine similarity and temporal consistency analysis.
"""

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from src.visual.visual_integrity import DocumentIntegrityClassifier

logger = logging.getLogger(__name__)


class DocumentFeatureExtractor:
    """
    Extracts deep visual embeddings from normalized document images.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = "models/weights/best_integrity_model.pth",
        backbone_name: str = "efficientnet_b0",
        device: Optional[Union[str, torch.device]] = None,
        image_size: Sequence[int] = (224, 224),
    ):
        self.backbone_name = backbone_name
        self.image_size = tuple(image_size)

        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Check if checkpoint exists
        ckpt_p = Path(checkpoint_path) if checkpoint_path else None
        valid_ckpt = ckpt_p is not None and ckpt_p.is_file()

        if valid_ckpt:
            logger.info(f"Loading feature extractor from checkpoint: {ckpt_p}")
            self.model = DocumentIntegrityClassifier(
                backbone_name=backbone_name,
                num_classes=2,
                checkpoint_path=ckpt_p,
                device=self.device,
            )
        else:
            logger.info(f"No checkpoint found at '{ckpt_p}'. Initializing feature extractor backbone.")
            self.model = DocumentIntegrityClassifier(
                backbone_name=backbone_name,
                num_classes=2,
                pretrained=True,
                device=self.device,
            )

        self.model.eval()
        self.embedding_dim = self.model.in_features

        self.transform = T.Compose([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _prepare_image(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> torch.Tensor:
        """Convert input to normalized PyTorch tensor (1, 3, H, W)."""
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("Received empty image array.")
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                rgb_img = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            else:
                rgb_img = image_input
            pil_img = Image.fromarray(rgb_img)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        return self.transform(pil_img).unsqueeze(0).to(self.device)

    def extract_frame(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Extract L2-normalized 1D feature embedding from a single frame.

        Returns:
            np.ndarray of shape (embedding_dim,)
        """
        tensor = self._prepare_image(image_input)
        with torch.no_grad():
            feat = self.model.extract_features(tensor).squeeze(0).cpu().numpy()

        if normalize:
            norm = np.linalg.norm(feat)
            if norm > 1e-8:
                feat = feat / norm
            else:
                feat = np.zeros_like(feat)

        return feat

    def extract_batch(
        self,
        images: Sequence[Union[str, Path, np.ndarray, Image.Image]],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Extract feature embeddings from a sequence of frames.

        Returns:
            np.ndarray of shape (N, embedding_dim)
        """
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        tensors = [self._prepare_image(img) for img in images]
        batch_tensor = torch.cat(tensors, dim=0)

        with torch.no_grad():
            feats = self.model.extract_features(batch_tensor).cpu().numpy()

        if normalize:
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            feats = feats / norms

        return feats
