"""
SAM2 Kalman Tracker - Wrapper around SAM2ObjectTracker (SAMURAI)

Provides a simplified API wrapper around SAMURAI's SAM2ObjectTracker
with Kalman filter for robust real-time tracking.

Reference: sam2_realtime (SAMURAI implementation)
Features:
- Dual memory banks (short-term + long-term)
- Kalman filter for motion prediction
- Occlusion handling
- Supports both manual selection and automatic detection
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from .utils import get_device
from .sam2_object_tracker import SAM2ObjectTracker


class SAM2KalmanTracker:
    """
    Wrapper around SAM2ObjectTracker for robust tracking with Kalman filter.

    This tracker uses SAMURAI's dual memory architecture with:
    - Short-term memory: Recent frames for quick adaptation
    - Long-term memory: Historical features for occlusion recovery
    - Kalman filter: Motion prediction for smooth tracking

    Supports both manual bbox selection and automatic detection.
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
            print(f"  NOTE: Uses Kalman filter for motion prediction and occlusion handling")

        # Build SAM2 Object Tracker using SAMURAI's builder
        try:
            from .build_sam_samurai import build_sam2_object_tracker
        except ImportError:
            raise ImportError(
                "SAMURAI build functions not found. Check installation."
            )

        # Build object tracker with Kalman filter
        # Note: We set SAMURAI's verbose to False to avoid cluttering output
        self.tracker = build_sam2_object_tracker(
            num_objects=num_objects,
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=str(self.device),
            verbose=False,  # Disable SAMURAI's verbose output
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
        Add a new object to track using Kalman filter.

        Args:
            frame: Current frame
            obj_id: Object ID (auto-assigned by tracker, this param is ignored)
            bbox: Bounding box as [x1, y1, x2, y2]
            points: Point prompts (alternative to bbox)
            labels: Point labels (for points input)

        Returns:
            Object ID assigned by the tracker
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")

        try:
            # Convert bbox to numpy array if provided
            box_array = None
            if bbox is not None:
                # SAMURAI expects box format: (2, 2) as [[x1, y1], [x2, y2]]
                # with normalized coordinates (0-1 range)
                # Input is [x1, y1, x2, y2] in pixel coordinates
                x1, y1, x2, y2 = bbox

                # Normalize by frame dimensions
                h, w = frame.shape[:2]
                x1_norm = x1 / w
                y1_norm = y1 / h
                x2_norm = x2 / w
                y2_norm = y2 / h

                box_array = np.array([[x1_norm, y1_norm], [x2_norm, y2_norm]], dtype=np.float32)

            # Get object ID before adding (SAMURAI uses curr_obj_idx counter)
            assigned_id = self.tracker.curr_obj_idx

            # Track new object using SAMURAI's method
            result = self.tracker.track_new_object(
                img=frame,
                points=points,
                box=box_array,
                mask=None
            )

            if self.verbose:
                prompt_type = "bbox" if bbox is not None else "points"
                print(f"Added object {assigned_id} with {prompt_type} to Kalman tracker")

            return assigned_id

        except Exception as e:
            if self.verbose:
                print(f"Error adding object: {e}")
                import traceback
                traceback.print_exc()
            return -1

    def track(
        self,
        frame: np.ndarray
    ) -> Tuple[int, List[int], Dict[int, np.ndarray]]:
        """
        Track objects in the current frame using Kalman filter.

        Args:
            frame: Current video frame (numpy array, BGR format)

        Returns:
            Tuple of (frame_idx, object_ids, masks_dict)
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")

        try:
            if self.verbose:
                print(f"DEBUG: Kalman track() called for frame_idx={self.frame_idx}")

            # Run tracking with Kalman filter on all objects
            prediction = self.tracker.track_all_objects(frame)

            if self.verbose:
                print(f"DEBUG: track_all_objects() returned")

            # Extract masks
            masks_dict = {}
            if "pred_masks" in prediction and prediction["pred_masks"] is not None:
                pred_masks = prediction["pred_masks"]

                # Handle different tensor shapes
                if pred_masks.ndim == 4:
                    # Shape: (1, num_objects, H, W) - squeeze batch dimension
                    pred_masks = pred_masks.squeeze(0)
                elif pred_masks.ndim == 3:
                    # Shape: (num_objects, H, W) - already correct
                    pass
                else:
                    if self.verbose:
                        print(f"Warning: Unexpected pred_masks shape: {pred_masks.shape}")
                    return self.frame_idx, [], {}

                # Convert each mask to numpy and add to dict
                num_objects = pred_masks.shape[0]
                for obj_idx in range(num_objects):
                    mask = pred_masks[obj_idx]

                    # Convert to numpy if needed
                    if hasattr(mask, 'cpu'):
                        mask = mask.cpu().numpy()

                    # Apply threshold
                    mask_binary = mask > 0.0

                    # Only include non-empty masks
                    if mask_binary.any():
                        masks_dict[obj_idx] = mask_binary

            obj_ids = list(masks_dict.keys())
            self.frame_idx += 1

            if self.verbose:
                if len(obj_ids) > 0:
                    print(f"Tracking {len(obj_ids)} objects with Kalman filter")
                else:
                    print(f"Warning: No masks returned from Kalman tracker")
                    if "pred_masks" in prediction:
                        print(f"  pred_masks exists but shape: {prediction['pred_masks'].shape if prediction['pred_masks'] is not None else None}")

            return self.frame_idx, obj_ids, masks_dict

        except Exception as e:
            if self.verbose:
                print(f"Tracking error: {e}")
                import traceback
                traceback.print_exc()
            return self.frame_idx, [], {}

    def reset(self):
        """Reset the tracker state."""
        # Reset SAMURAI's object counter
        self.tracker.curr_obj_idx = 0
        self.is_initialized = False
        self.frame_idx = 0
        if self.verbose:
            print("Tracker reset")

    def get_tracked_objects(self) -> List[int]:
        """Get list of currently tracked object IDs."""
        if not self.is_initialized:
            return []

        # Get tracked object IDs from curr_obj_idx counter
        try:
            # curr_obj_idx tracks how many objects have been added
            num_objects = self.tracker.curr_obj_idx
            # Return 0-based indices as object IDs
            return list(range(num_objects))
        except:
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
