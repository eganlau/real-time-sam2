"""
SAM2 Camera Tracker - Wrapper around SAM2CameraPredictor

Provides a simplified API wrapper around Meta's SAM2CameraPredictor
for real-time streaming object tracking.

Reference: segment-anything-2-real-time
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from .utils import get_device


class SAM2CameraTracker:
    """
    Simplified wrapper around SAM2CameraPredictor for streaming tracking.

    This provides our standard API (initialize, add_object, track, reset)
    around Meta's SAM2CameraPredictor implementation.
    """

    def __init__(
        self,
        config_file: str,
        checkpoint_path: str,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the SAM2 camera tracker.

        Args:
            config_file: Path to SAM2 config file
            checkpoint_path: Path to SAM2 checkpoint
            device: Device to use ('mps', 'cuda', 'cpu')
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
            print(f"Loading SAM2 camera tracker...")
            print(f"  Config: {config_file}")
            print(f"  Checkpoint: {checkpoint_path}")
            print(f"  Device: {self.device}")

        # Import build function for SAM2CameraPredictor
        try:
            from .build_sam_camera import build_sam2_camera_predictor
        except ImportError:
            raise ImportError(
                "Camera predictor build functions not found. Check installation."
            )

        # Build SAM2 camera predictor using proper builder
        self.predictor = build_sam2_camera_predictor(
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=str(self.device),
            mode="eval",
            apply_postprocessing=True
        )

        # Tracking state
        self.is_initialized = False
        self.frame_idx = 0
        self.next_obj_id = 1

        if verbose:
            print("SAM2 camera tracker initialized!")

    def initialize(self, first_frame: np.ndarray) -> bool:
        """
        Initialize tracking with the first frame.

        Args:
            first_frame: First video frame (numpy array, BGR format)

        Returns:
            True if initialization successful
        """
        try:
            # Load first frame into predictor
            self.predictor.load_first_frame(first_frame)

            self.is_initialized = True
            self.frame_idx = 0

            if self.verbose:
                print(f"Initialized with frame size: {first_frame.shape[1]}x{first_frame.shape[0]}")

            return True

        except Exception as e:
            if self.verbose:
                print(f"Error initializing tracker: {e}")
                import traceback
                traceback.print_exc()
            return False

    def add_object(
        self,
        frame: np.ndarray,
        obj_id: Optional[int] = None,
        bbox: Optional[List[int]] = None,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None
    ) -> int:
        """
        Add a new object to track.

        Args:
            frame: Current frame (not used for camera predictor)
            obj_id: Object ID (None = auto-assign)
            bbox: Bounding box as [x1, y1, x2, y2]
            points: Point prompts as (N, 2) array
            labels: Point labels (1=foreground, 0=background)

        Returns:
            The object ID assigned
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")

        if obj_id is None:
            obj_id = self.next_obj_id
            self.next_obj_id += 1

        try:
            # Check if tracking has already started
            tracking_started = self.predictor.condition_state.get("tracking_has_started", False)

            if tracking_started:
                # SAM2CameraPredictor doesn't support adding new objects after tracking starts
                # User must reset (press 'R') to add more objects
                if self.verbose:
                    print(f"Cannot add object {obj_id}: tracking already started.")
                    print("Press 'R' to reset and add multiple objects before tracking starts.")
                raise RuntimeError(
                    "Cannot add new objects after tracking starts. "
                    "Press 'R' to reset, then select all objects before they start moving."
                )

            # Add prompt before tracking starts
            self.predictor.add_new_prompt(
                frame_idx=self.frame_idx,
                obj_id=obj_id,
                bbox=bbox if bbox is not None else None,
                points=points,
                labels=labels,
                normalize_coords=True
            )

            if self.verbose:
                prompt_type = "bbox" if bbox is not None else "points"
                print(f"Added object {obj_id} with {prompt_type} prompt")

            return obj_id

        except Exception as e:
            if self.verbose:
                print(f"Error adding object {obj_id}: {e}")
            raise

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

        try:
            if self.verbose:
                print(f"DEBUG: Calling predictor.track() for frame_idx={self.frame_idx}")
                print(f"DEBUG: Frame shape={frame.shape}, dtype={frame.dtype}")
                print(f"DEBUG: Tracked objects={self.get_tracked_objects()}")

            # Track in current frame
            obj_ids, video_res_masks = self.predictor.track(frame)

            if self.verbose:
                print(f"DEBUG: Track returned obj_ids={obj_ids}")
                if video_res_masks is not None:
                    print(f"DEBUG: Masks shape={video_res_masks.shape}, dtype={video_res_masks.dtype}")
                else:
                    print(f"DEBUG: No masks returned")

            # Convert to masks dictionary
            masks_dict = {}
            if video_res_masks is not None:
                for i, obj_id in enumerate(obj_ids):
                    # video_res_masks shape: (num_objects, H, W)
                    mask = video_res_masks[i] > 0.0
                    masks_dict[obj_id] = mask

            self.frame_idx += 1

            return self.frame_idx, obj_ids, masks_dict

        except Exception as e:
            if self.verbose:
                print(f"ERROR in track(): {e}")
                import traceback
                traceback.print_exc()
            return self.frame_idx, [], {}

    def reset(self):
        """Reset the tracker state."""
        self.predictor.reset_state()
        self.is_initialized = False
        self.frame_idx = 0
        if self.verbose:
            print("Tracker reset")

    def get_tracked_objects(self) -> List[int]:
        """Get list of currently tracked object IDs."""
        if not self.is_initialized:
            return []
        return self.predictor.condition_state.get("obj_ids", [])

    def remove_object(self, obj_id: int):
        """
        Remove an object from tracking.

        Args:
            obj_id: Object ID to remove
        """
        # SAM2CameraPredictor doesn't have explicit remove method
        # This would need to be implemented by clearing from condition_state
        if self.verbose:
            print(f"Warning: remove_object not fully implemented for SAM2CameraPredictor")

    def __repr__(self) -> str:
        status = "initialized" if self.is_initialized else "not initialized"
        num_objects = len(self.get_tracked_objects())
        return (
            f"SAM2CameraTracker(device={self.device}, {status}, "
            f"tracking {num_objects} objects, frame {self.frame_idx})"
        )
