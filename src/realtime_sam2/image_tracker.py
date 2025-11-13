"""
Simple SAM2 Image-based Tracker for Real-Time Use

This uses SAM2ImagePredictor which works frame-by-frame,
better suited for real-time webcam tracking than VideoPredictor.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from .utils import get_device


class SAM2ImageTracker:
    """
    Real-time object tracker using SAM2's image predictor.

    This is simpler and more suitable for real-time webcam use
    than the video predictor.
    """

    def __init__(
        self,
        config_file: str,
        checkpoint_path: str,
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = True
    ):
        """
        Initialize the SAM2 image tracker.

        Args:
            config_file: Path to SAM2 config file (e.g., 'configs/sam2.1/sam2.1_hiera_t.yaml')
            checkpoint_path: Path to SAM2 checkpoint file
            device: Device to use ('mps', 'cuda', 'cpu', or torch.device)
            verbose: Print initialization info
        """
        self.verbose = verbose

        # Import SAM2
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            raise ImportError(
                "SAM2 is not installed. Please install it from "
                "https://github.com/facebookresearch/sam2"
            )

        # Set device
        if device is None:
            self.device = get_device(prefer_mps=True, verbose=verbose)
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        if verbose:
            print(f"Loading SAM2 image predictor...")
            print(f"  Config: {config_file}")
            print(f"  Checkpoint: {checkpoint_path}")
            print(f"  Device: {self.device}")

        # Build model
        sam2_model = build_sam2(config_file, checkpoint_path, device=str(self.device))
        self.predictor = SAM2ImagePredictor(sam2_model)

        # Tracking state
        self.is_initialized = False
        self.tracked_objects = {}  # obj_id -> prompt_data
        self.next_obj_id = 1

        # Tracking parameters
        self.bbox_smoothing = 0.7  # Weight for previous bbox (0.7 = 70% old, 30% new)
        self.bbox_expansion = 1.1  # Expand bbox by 10% to prevent shrinkage
        self.min_bbox_size = 20  # Minimum width/height in pixels

        if verbose:
            print("SAM2 image tracker initialized!")

    def initialize(self, first_frame: np.ndarray) -> bool:
        """
        Initialize tracking with the first frame.

        Args:
            first_frame: First video frame (numpy array, BGR format)

        Returns:
            True if initialization successful
        """
        self.is_initialized = True
        return True

    def add_object(
        self,
        frame: np.ndarray,
        obj_id: Optional[int] = None,
        bbox: Optional[Union[List[int], Tuple[int, int, int, int]]] = None,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None
    ) -> int:
        """
        Add a new object to track.

        Args:
            frame: Current frame
            obj_id: Object ID (None = auto-assign)
            bbox: Bounding box as [x1, y1, x2, y2]
            points: Point prompts as (N, 2) array
            labels: Point labels (1=foreground, 0=background)

        Returns:
            The object ID assigned
        """
        if obj_id is None:
            obj_id = self.next_obj_id
            self.next_obj_id += 1

        # Store prompt for this object
        self.tracked_objects[obj_id] = {
            'bbox': np.array(bbox) if bbox is not None else None,
            'points': points,
            'labels': labels if labels is not None else (
                np.ones(len(points), dtype=np.int32) if points is not None else None
            )
        }

        if self.verbose:
            prompt_type = "bbox" if bbox is not None else "points"
            print(f"Added object {obj_id} with {prompt_type} prompt")

        return obj_id

    def _mask_to_bbox(self, mask: np.ndarray, expand: bool = True) -> Optional[np.ndarray]:
        """
        Extract bounding box from a binary mask with optional expansion.

        Args:
            mask: Binary mask array
            expand: Whether to expand the bbox to prevent shrinkage

        Returns:
            Bounding box as [x1, y1, x2, y2] or None if mask is empty
        """
        if not mask.any():
            return None

        # Find non-zero pixels
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            return None

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

        # Expand bbox to prevent gradual shrinkage
        if expand:
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            new_width = width * self.bbox_expansion
            new_height = height * self.bbox_expansion

            bbox = np.array([
                center_x - new_width / 2,
                center_y - new_height / 2,
                center_x + new_width / 2,
                center_y + new_height / 2
            ], dtype=np.float32)

            # Clip to image bounds
            h, w = mask.shape[:2]
            bbox = np.clip(bbox, [0, 0, 0, 0], [w, h, w, h])

        return bbox

    def _smooth_bbox(self, new_bbox: np.ndarray, old_bbox: np.ndarray) -> np.ndarray:
        """
        Smooth bounding box using exponential moving average.

        Args:
            new_bbox: Newly detected bbox [x1, y1, x2, y2]
            old_bbox: Previous bbox [x1, y1, x2, y2]

        Returns:
            Smoothed bbox
        """
        # Ensure minimum size
        width = new_bbox[2] - new_bbox[0]
        height = new_bbox[3] - new_bbox[1]

        if width < self.min_bbox_size or height < self.min_bbox_size:
            # Keep old bbox if new one is too small
            return old_bbox

        # Exponential moving average
        smoothed = self.bbox_smoothing * old_bbox + (1 - self.bbox_smoothing) * new_bbox

        return smoothed.astype(np.float32)

    def track(
        self,
        frame: np.ndarray
    ) -> Tuple[int, List[int], Dict[int, np.ndarray]]:
        """
        Track objects in the current frame.

        Args:
            frame: Current video frame (numpy array, BGR format)

        Returns:
            Tuple of (frame_idx, object_ids, masks_dict)
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")

        if len(self.tracked_objects) == 0:
            return 0, [], {}

        # Set the image for prediction
        with torch.inference_mode():
            self.predictor.set_image(frame)

            masks_dict = {}

            # Get mask for each tracked object
            for obj_id, prompt_data in self.tracked_objects.items():
                try:
                    masks, scores, logits = self.predictor.predict(
                        point_coords=prompt_data['points'],
                        point_labels=prompt_data['labels'],
                        box=prompt_data['bbox'],
                        multimask_output=False
                    )

                    # Use the first (and only) mask
                    if masks is not None and len(masks) > 0:
                        mask = masks[0]
                        masks_dict[obj_id] = mask

                        # Update bbox based on current mask for next frame
                        # This enables actual tracking as objects move
                        new_bbox = self._mask_to_bbox(mask, expand=True)
                        if new_bbox is not None:
                            old_bbox = prompt_data['bbox']
                            # Smooth bbox to reduce jitter and prevent shrinkage
                            smoothed_bbox = self._smooth_bbox(new_bbox, old_bbox)
                            self.tracked_objects[obj_id]['bbox'] = smoothed_bbox

                except Exception as e:
                    if self.verbose:
                        print(f"Error predicting for object {obj_id}: {e}")

        return 0, list(self.tracked_objects.keys()), masks_dict

    def reset(self):
        """Reset the tracker state."""
        self.tracked_objects = {}
        self.next_obj_id = 1
        if self.verbose:
            print("Tracker reset")

    def get_tracked_objects(self) -> List[int]:
        """Get list of currently tracked object IDs."""
        return sorted(list(self.tracked_objects.keys()))

    def remove_object(self, obj_id: int):
        """Remove an object from tracking."""
        if obj_id in self.tracked_objects:
            del self.tracked_objects[obj_id]
            if self.verbose:
                print(f"Removed object {obj_id}")

    def __repr__(self) -> str:
        status = "initialized" if self.is_initialized else "not initialized"
        return (
            f"SAM2ImageTracker(device={self.device}, "
            f"{status}, tracking {len(self.tracked_objects)} objects)"
        )
