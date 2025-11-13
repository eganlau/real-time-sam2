"""
Object Detector - Optional YOLO integration for automatic object detection.

This module provides automatic object detection capabilities using YOLOv8
to bootstrap SAM2 tracking without manual annotation.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np


class ObjectDetector:
    """
    Object detector using YOLOv8 for automatic detection.

    This class provides automatic detection of specific object classes
    (e.g., "cell phone") to bootstrap SAM2 tracking.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        target_classes: Optional[List[str]] = None,
        device: str = "cpu",
        verbose: bool = True
    ):
        """
        Initialize the object detector.

        Args:
            model_name: YOLO model name (e.g., 'yolov8n.pt', 'yolov8s.pt')
            confidence_threshold: Minimum confidence for detections
            target_classes: List of class names to detect (None = all classes)
            device: Device to use ('cpu', 'cuda', 'mps')
            verbose: Print initialization info
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes
        self.device = device
        self.verbose = verbose
        self.model = None
        self.class_names = None

        self._load_model()

    def _load_model(self):
        """Load the YOLO model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Ultralytics YOLO is not installed. Install with: "
                "pip install ultralytics"
            )

        try:
            if self.verbose:
                print(f"Loading YOLO model: {self.model_name}...")

            self.model = YOLO(self.model_name)
            self.class_names = self.model.names

            if self.verbose:
                print(f"YOLO model loaded with {len(self.class_names)} classes")
                if self.target_classes:
                    print(f"Filtering for classes: {self.target_classes}")

        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def detect(
        self,
        frame: np.ndarray,
        return_format: str = "bbox"
    ) -> List[Dict]:
        """
        Detect objects in a frame.

        Args:
            frame: Input frame (BGR format)
            return_format: Format of detection output ('bbox', 'points', 'mask')

        Returns:
            List of detection dictionaries with keys:
            - 'class_name': Object class name
            - 'confidence': Detection confidence score
            - 'bbox': Bounding box as [x1, y1, x2, y2]
            - 'points': (Optional) Center point as [x, y]
            - 'mask': (Optional) Segmentation mask if available
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False
        )

        detections = []

        # Process results
        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Get box data
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = self.class_names[cls_id]

                # Filter by target classes if specified
                if self.target_classes and class_name not in self.target_classes:
                    continue

                # Create detection dictionary
                detection = {
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox': [int(x) for x in xyxy]
                }

                # Add center point if requested
                if return_format in ['points', 'both']:
                    x1, y1, x2, y2 = xyxy
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    detection['points'] = np.array([[cx, cy]])

                # Add mask if available and requested
                if return_format in ['mask', 'both'] and hasattr(result, 'masks') and result.masks is not None:
                    # YOLOv8 segmentation model provides masks
                    mask_idx = boxes.id == box.id if hasattr(boxes, 'id') else None
                    if mask_idx is not None:
                        detection['mask'] = result.masks[mask_idx].data.cpu().numpy()

                detections.append(detection)

        if self.verbose and detections:
            print(f"Detected {len(detections)} objects: {[d['class_name'] for d in detections]}")

        return detections

    def detect_phones(
        self,
        frame: np.ndarray,
        return_format: str = "bbox"
    ) -> List[Dict]:
        """
        Convenience method to detect cell phones specifically.

        Args:
            frame: Input frame (BGR format)
            return_format: Format of detection output

        Returns:
            List of phone detections
        """
        # Temporarily set target classes to phone-related classes
        original_target = self.target_classes
        self.target_classes = ["cell phone"]

        detections = self.detect(frame, return_format)

        # Restore original target classes
        self.target_classes = original_target

        return detections

    def filter_by_area(
        self,
        detections: List[Dict],
        min_area: Optional[int] = None,
        max_area: Optional[int] = None
    ) -> List[Dict]:
        """
        Filter detections by bounding box area.

        Args:
            detections: List of detection dictionaries
            min_area: Minimum bbox area in pixels
            max_area: Maximum bbox area in pixels

        Returns:
            Filtered list of detections
        """
        filtered = []

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            area = (x2 - x1) * (y2 - y1)

            if min_area is not None and area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue

            filtered.append(det)

        return filtered

    def non_max_suppression(
        self,
        detections: List[Dict],
        iou_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Apply non-maximum suppression to remove duplicate detections.

        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for suppression

        Returns:
            Filtered list of detections
        """
        if not detections:
            return []

        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)

        keep = []

        while detections:
            # Keep the highest confidence detection
            current = detections.pop(0)
            keep.append(current)

            # Remove detections with high IoU
            detections = [
                det for det in detections
                if self._calculate_iou(current['bbox'], det['bbox']) < iou_threshold
            ]

        return keep

    def _calculate_iou(
        self,
        bbox1: List[int],
        bbox2: List[int]
    ) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            bbox1: First bbox as [x1, y1, x2, y2]
            bbox2: Second bbox as [x1, y1, x2, y2]

        Returns:
            IoU value (0-1)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def __repr__(self) -> str:
        target = f", target={self.target_classes}" if self.target_classes else ""
        return (
            f"ObjectDetector(model={self.model_name}, "
            f"conf={self.confidence_threshold}{target})"
        )
