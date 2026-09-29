"""
Unit and Integration Tests for MIDV-500 Document Segmentation Experiment
========================================================================
Tests the isolated experimental learned segmentation pipeline:
- MIDV500Segmenter initialization & weight loading
- Preprocessing and tensor dimension formatting
- Prediction of probability maps and masks
- Polygonal quadrilateral derivation and canonical point ordering
- Graceful handling of blank/empty inputs
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from scripts.experiments.evaluate_midv500_segmentation import MIDV500Segmenter
from scripts.validate_public_preprocessing import compute_corner_errors, compute_quadrilateral_iou
from src.preprocessing.perspective_corrector import order_points


@pytest.fixture(scope="module")
def segmenter():
    weights_path = Path("models/weights/midv500_unet_resnet34.pth")
    if not weights_path.exists():
        pytest.skip(f"MIDV-500 weights not present at {weights_path}")
    return MIDV500Segmenter(weights_path=str(weights_path), input_size=512, device="cpu")


def test_segmenter_initialization(segmenter):
    """Verify segmenter initializes correctly with eval mode and correct normalization parameters."""
    assert segmenter.model is not None
    assert not segmenter.model.training
    assert segmenter.input_size == 512
    assert segmenter.mean.shape == (3,)
    assert segmenter.std.shape == (3,)


def test_segmenter_preprocessing(segmenter):
    """Verify image preprocessing resizes, pads to multiple of 32, and outputs valid torch tensor."""
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_img[100:380, 150:490] = 200

    tensor, nh, nw, pad_h, pad_w = segmenter.preprocess_image(dummy_img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.ndim == 4
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == 3
    # Check divisible by 32
    assert tensor.shape[2] % 32 == 0
    assert tensor.shape[3] % 32 == 0
    assert pad_h >= 0
    assert pad_w >= 0


def test_segmenter_predict_mask_properties(segmenter):
    """Verify mask prediction produces valid probability range and uint8 binary mask."""
    dummy_img = np.zeros((400, 600, 3), dtype=np.uint8)
    # Draw a synthetic white rectangle resembling a document card
    dummy_img[80:320, 120:480] = 220

    mask, prob_map, inf_time_ms = segmenter.predict_mask(dummy_img)

    assert mask.shape == (400, 600)
    assert prob_map.shape == (400, 600)
    assert mask.dtype == np.uint8
    assert prob_map.dtype == np.float32
    assert np.all(prob_map >= 0.0) and np.all(prob_map <= 1.0)
    assert set(np.unique(mask)).issubset({0, 255})
    assert inf_time_ms > 0.0


def test_segmenter_extract_quadrilateral_synthetic_card(segmenter):
    """Verify extracting 4 corners from a clean synthetic rectangular mask."""
    h, w = 480, 640
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[100:380, 150:500] = 255
    prob_map = (mask / 255.0).astype(np.float32)

    detected, corners, conf, method = segmenter.extract_quadrilateral(mask, prob_map)

    assert detected is True
    assert corners is not None
    assert corners.shape == (4, 2)
    assert conf > 0.9
    assert "poly_approx" in method or "rect" in method

    # Verify canonical ordering: TL, TR, BR, BL
    tl, tr, br, bl = corners
    assert tl[0] <= tr[0]
    assert bl[0] <= br[0]
    assert tl[1] <= bl[1]
    assert tr[1] <= br[1]


def test_segmenter_empty_frame_handling(segmenter):
    """Verify graceful failure on pure black image with no document content."""
    black_img = np.zeros((300, 400, 3), dtype=np.uint8)
    res = segmenter.process_frame(black_img)

    assert isinstance(res, dict)
    assert "detected" in res
    assert "corners" in res
    assert "inference_time_ms" in res
    # Either not detected or area too small
    if not res["detected"]:
        assert res["corners"] is None
        assert res["confidence"] == 0.0


def test_quadrilateral_metrics_with_segmenter_corners():
    """Verify IoU and corner error calculations between synthetic ground truth and predicted corners."""
    gt_corners = np.array([
        [100.0, 100.0],
        [400.0, 100.0],
        [400.0, 300.0],
        [100.0, 300.0]
    ], dtype=np.float32)

    # Perfect match
    iou_perfect = compute_quadrilateral_iou(gt_corners, gt_corners)
    mean_err, med_err, max_err, _ = compute_corner_errors(gt_corners, gt_corners)
    assert pytest.approx(iou_perfect, 1e-4) == 1.0
    assert pytest.approx(mean_err, 1e-4) == 0.0

    # Slight 5px shift
    pred_shifted = gt_corners + 5.0
    iou_shifted = compute_quadrilateral_iou(pred_shifted, gt_corners)
    mean_err_s, _, _, _ = compute_corner_errors(pred_shifted, gt_corners)
    assert iou_shifted > 0.90
    assert pytest.approx(mean_err_s, 0.1) == np.linalg.norm([5.0, 5.0])
