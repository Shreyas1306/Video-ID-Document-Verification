"""
Unit and Integration Tests for Document Detection (Phase 2)
===========================================================
Tests:
- Invalid image inputs (None, empty, incorrect dimensionality)
- Valid frame input structure and type contracts
- No detection handling (blank frame / high threshold)
- Successful detection parsing, coordinate clamping, and cropping
- DocumentCropper padding, boundary safety, and debug annotations
"""

from pathlib import Path
from unittest.mock import MagicMock
import cv2
import numpy as np
import pytest
import torch

from src.preprocessing.document_cropper import DocumentCropper
from src.preprocessing.document_detector import YOLODocumentDetector, run_detection_pipeline


@pytest.fixture(scope="module")
def detector() -> YOLODocumentDetector:
    """Instantiate a real YOLO detector instance."""
    return YOLODocumentDetector(model_name_or_path="yolov8n.pt", confidence_threshold=0.25)


class TestDocumentDetectorInputs:
    """Test error handling on invalid image inputs."""

    def test_none_input_raises_valueerror(self, detector: YOLODocumentDetector):
        with pytest.raises(ValueError, match="Invalid frame input"):
            detector.detect_frame(None)

    def test_empty_array_raises_valueerror(self, detector: YOLODocumentDetector):
        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Invalid image array dimensions"):
            detector.detect_frame(empty_frame)

    def test_2d_grayscale_raises_valueerror(self, detector: YOLODocumentDetector):
        gray_frame = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="Invalid image array dimensions"):
            detector.detect_frame(gray_frame)

    def test_non_numpy_input_raises_valueerror(self, detector: YOLODocumentDetector):
        with pytest.raises(ValueError, match="Invalid frame input"):
            detector.detect_frame("not_an_image")


class TestDocumentDetectorOutputs:
    """Test detection outputs on valid images."""

    def test_valid_frame_structure(self, detector: YOLODocumentDetector):
        """Verify output dictionary contract on valid frame input."""
        frame = np.full((320, 480, 3), 120, dtype=np.uint8)
        result = detector.detect_frame(frame, frame_id="frame_001")

        assert isinstance(result, dict)
        assert result["frame_id"] == "frame_001"
        assert "detected" in result
        assert "bbox" in result
        assert "confidence" in result
        assert "class_name" in result
        assert "class_id" in result
        assert "all_detections" in result

    def test_no_detection_case(self, detector: YOLODocumentDetector):
        """Verify no-detection behavior on flat image with high confidence threshold."""
        frame = np.full((300, 300, 3), 180, dtype=np.uint8)
        # Use high threshold to guarantee no spurious COCO detections on solid background
        det_high = YOLODocumentDetector(confidence_threshold=0.99)
        result = det_high.detect_frame(frame, frame_id="flat_test")

        assert result["detected"] is False
        assert result["bbox"] is None
        assert result["confidence"] == 0.0
        assert result["class_name"] is None
        assert result["class_id"] is None
        assert result["all_detections"] == []

    def test_successful_detection_simulation(self):
        """Verify behavior when detector receives a valid detection."""
        detector_mock = YOLODocumentDetector(confidence_threshold=0.2)

        # Mock results from YOLO model
        mock_result = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 1
        mock_boxes.xyxy = torch.tensor([[50.2, 40.1, 250.7, 180.9]])
        mock_boxes.conf = torch.tensor([0.88])
        mock_boxes.cls = torch.tensor([0.0])
        mock_result.boxes = mock_boxes
        mock_result.names = {0: "mock_document"}

        detector_mock.model.predict = MagicMock(return_value=[mock_result])

        frame = np.full((300, 400, 3), 200, dtype=np.uint8)
        res = detector_mock.detect_frame(frame, frame_id="sim_01")

        assert res["detected"] is True
        assert res["bbox"] == [50, 40, 251, 181]
        assert res["confidence"] == 0.88
        assert res["class_name"] == "mock_document"
        assert len(res["all_detections"]) == 1


class TestDocumentCropper:
    """Test DocumentCropper functionality and debug visualizer."""

    def test_crop_valid_region(self):
        cropper = DocumentCropper(default_padding=0.0)
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        # Put red square in the bbox
        frame[50:150, 60:180] = (0, 0, 255)

        bbox = [60, 50, 180, 150]
        crop = cropper.crop_region(frame, bbox)

        assert crop is not None
        assert crop.shape == (100, 120, 3)
        assert np.all(crop == (0, 0, 255))

    def test_crop_boundary_clamping(self):
        cropper = DocumentCropper(default_padding=0.5)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Box near edge
        bbox = [5, 5, 95, 95]
        crop = cropper.crop_region(frame, bbox)

        assert crop is not None
        # Padded box should not exceed frame boundaries (100, 100)
        assert crop.shape[0] <= 100
        assert crop.shape[1] <= 100

    def test_crop_invalid_box_returns_none(self):
        cropper = DocumentCropper()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        assert cropper.crop_region(frame, None) is None
        assert cropper.crop_region(frame, [100, 100, 50, 50]) is None  # inverted box
        assert cropper.crop_region(frame, [10, 20]) is None  # malformed

    def test_draw_annotation(self):
        cropper = DocumentCropper()
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        bbox = [20, 20, 120, 120]

        annotated = cropper.draw_annotation(frame, bbox, confidence=0.85, class_name="document")
        assert annotated.shape == frame.shape
        # Visual bounding box should have altered pixels from zero
        assert np.any(annotated > 0)
