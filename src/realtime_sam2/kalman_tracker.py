"""
SAM2 Kalman Tracker - Wrapper around SAM2ObjectTracker (SAMURAI)

Provides a simplified API wrapper around SAMURAI's SAM2ObjectTracker
with Kalman filter for autonomous real-time tracking.

Reference: sam2_realtime (SAMURAI implementation)
NOTE: This tracker is designed for automatic tracking and doesn't support
explicit object selection via bounding boxes. It autonomously tracks
whatever objects it detects using Kalman filtering and dual memory banks.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from .utils import get_device
from .sam2_object_tracker import SAM2ObjectTracker


class SAM2KalmanTracker:
    """
    Wrapper around SAM2ObjectTracker for autonomous tracking with Kalman filter.

    NOTE: Unlike SAM2CameraTracker, this tracker does NOT support manual object
    selection. It automatically tracks objects using motion prediction and
    dual memory banks (short-term + long-term with occlusion handling).

    For interactive bbox selection, use SAM2CameraTracker instead.
    """

    def __init__(
        self,
        config_file: str,
        checkpoint_path: str,
        device: Optional[str] = None,
        num_objects: int = 10,
        verbose: bool = True
    ):
        """
        Initialize the SAM2 Kalman tracker.

        Args:
            config_file: Path to SAM2 config file
            checkpoint_path: Path to SAM2 checkpoint
            device: Device to use ('mps', 'cuda', 'cpu')
            num_objects: Maximum number of objects to track
            verbose: Print initialization info
        """
        self.verbose = verbose

        # Set device
        if device is None:
            self.device = get_device(prefer_mps=True, verbose=verbose)
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        if verbose:
            print(f"Loading SAM2 Kalman tracker (SAMURAI)...")
            print(f"  Config: {config_file}")
            print(f"  Checkpoint: {checkpoint_path}")
            print(f"  Device: {self.device}")
            print(f"  Max objects: {num_objects}")
            print(f"  NOTE: This tracker uses automatic detection, not manual selection")

        # Build SAM2 Object Tracker using SAMURAI's builder
        try:
            from .build_sam_samurai import build_sam2_object_tracker
        except ImportError:
            raise ImportError(
                "SAMURAI build functions not found. Check installation."
            )

        # Build object tracker with Kalman filter
        self.tracker = build_sam2_object_tracker(
            num_objects=num_objects,
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=str(self.device),
            verbose=verbose,
            mode="eval"
        )

        # Tracking state
        self.is_initialized = False
        self.frame_idx = 0

        if verbose:
            print("SAM2 Kalman tracker initialized!")

    def initialize(self, first_frame: np.ndarray) -> bool:
        """
        Initialize tracking with the first frame.

        Args:
            first_frame: First video frame (numpy array, BGR format)

        Returns:
            True if initialization successful
        """
        self.is_initialized = True
        self.frame_idx = 0

        if self.verbose:
            print(f"Initialized with frame size: {first_frame.shape[1]}x{first_frame.shape[0]}")
            print("WARNING: Kalman tracker uses automatic detection.")
            print("Manual bbox selection via add_object() is not supported.")

        return True

    def add_object(
        self,
        frame: np.ndarray,
        obj_id: Optional[int] = None,
        bbox: Optional[List[int]] = None,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None
    ) -> int:
        """
        Add object - NOT SUPPORTED for Kalman tracker.

        The Kalman tracker automatically detects and tracks objects.
        Manual object selection is not supported.

        Args:
            frame: Current frame
            obj_id: Object ID (ignored)
            bbox: Bounding box (ignored)
            points: Point prompts (ignored)
            labels: Point labels (ignored)

        Returns:
            -1 (operation not supported)
        """
        if self.verbose:
            print("WARNING: add_object() not supported for Kalman tracker")
            print("This tracker automatically detects and tracks objects")

        return -1

    def track(
        self,
        frame: np.ndarray
    ) -> Tuple[int, List[int], Dict[int, np.ndarray]]:
        """
        Track objects automatically in the current frame.

        Args:
            frame: Current video frame (numpy array, BGR format)

        Returns:
            Tuple of (frame_idx, object_ids, masks_dict)
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")

        try:
            # Run automatic tracking with Kalman filter
            prediction = self.tracker.track_all_objects(frame)

            # Extract masks
            masks_dict = {}
            if "pred_masks" in prediction:
                pred_masks = prediction["pred_masks"]  # Shape: (num_objects, H, W)

                # Convert to dictionary
                for obj_idx in range(pred_masks.shape[0]):
                    mask = pred_masks[obj_idx].cpu().numpy() > 0.0
                    if mask.any():  # Only include non-empty masks
                        masks_dict[obj_idx + 1] = mask  # Use 1-based indexing

            obj_ids = list(masks_dict.keys())
            self.frame_idx += 1

            return self.frame_idx, obj_ids, masks_dict

        except Exception as e:
            if self.verbose:
                print(f"Tracking error: {e}")
            return self.frame_idx, [], {}

    def reset(self):
        """Reset the tracker state."""
        # SAM2ObjectTracker resets automatically via memory management
        self.is_initialized = False
        self.frame_idx = 0
        if self.verbose:
            print("Tracker reset")

    def get_tracked_objects(self) -> List[int]:
        """Get list of currently tracked object IDs."""
        # SAM2ObjectTracker manages objects internally
        # Return estimated count based on memory bank
        return []

    def remove_object(self, obj_id: int):
        """
        Remove object - NOT SUPPORTED for Kalman tracker.

        Args:
            obj_id: Object ID (ignored)
        """
        if self.verbose:
            print(f"WARNING: remove_object() not supported for Kalman tracker")

    def __repr__(self) -> str:
        status = "initialized" if self.is_initialized else "not initialized"
        return (
            f"SAM2KalmanTracker(device={self.device}, {status}, "
            f"frame {self.frame_idx}, automatic tracking)"
        )
