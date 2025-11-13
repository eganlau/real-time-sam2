"""
SAM2 Camera Tracker - Real-time object tracking with streaming predictor.

This module provides a wrapper around SAM2's camera predictor optimized
for streaming video from webcams or video files.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from .utils import get_device, warmup_model


class SAM2CameraTracker:
    """
    Real-time object tracker using SAM2's camera predictor.

    This class wraps SAM2's build_sam2_camera_predictor for efficient
    streaming video segmentation and tracking.
    """

    def __init__(
        self,
        config_file: str,
        checkpoint_path: str,
        device: Optional[Union[str, torch.device]] = None,
        use_compile: bool = True,
        use_bfloat16: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the SAM2 camera tracker.

        Args:
            config_file: Path to SAM2 configuration file
            checkpoint_path: Path to SAM2 checkpoint file
            device: Device to use ('mps', 'cuda', 'cpu', or torch.device)
            use_compile: Enable torch.compile for speedup
            use_bfloat16: Use bfloat16 mixed precision
            verbose: Print initialization info
        """
        self.verbose = verbose

        # Import SAM2 (deferred to allow installation without SAM2)
        try:
            from sam2.build_sam import build_sam2_video_predictor
            from sam2.sam2_video_predictor import SAM2VideoPredictor
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
            print(f"Loading SAM2 camera predictor...")
            print(f"  Config: {config_file}")
            print(f"  Checkpoint: {checkpoint_path}")
            print(f"  Device: {self.device}")

        # Build predictor
        self.predictor = build_sam2_video_predictor(
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=str(self.device)
        )

        # Streaming state
        self.inference_state = None
        self.frame_buffer = []

        # Enable torch.compile for speedup
        if use_compile:
            if verbose:
                print("Enabling torch.compile (will warmup on first use)...")
            try:
                self.predictor = torch.compile(
                    self.predictor,
                    mode="reduce-overhead"
                )
                self.is_compiled = True
            except Exception as e:
                if verbose:
                    print(f"Warning: torch.compile failed: {e}")
                self.is_compiled = False
        else:
            self.is_compiled = False

        self.use_bfloat16 = use_bfloat16
        self.is_initialized = False
        self.current_frame_idx = 0
        self.object_ids = set()
        self.next_obj_id = 1

        if verbose:
            print("SAM2 camera tracker initialized!")

    def warmup(self, dummy_frame: np.ndarray, num_frames: int = 3):
        """
        Warm up the model for torch.compile optimization.

        Args:
            dummy_frame: A representative frame for warmup
            num_frames: Number of warmup iterations
        """
        if self.is_compiled and self.verbose:
            print(f"Warming up compiled model with {num_frames} iterations...")

            try:
                # Create dummy frames
                dummy_frames = [dummy_frame] * num_frames

                # Initialize state
                with torch.inference_mode():
                    dummy_state = self.predictor.init_state(
                        video_path=dummy_frames,
                        offload_video_to_cpu=False,
                        offload_state_to_cpu=False,
                        async_loading_frames=False
                    )

                    # Add a dummy object
                    h, w = dummy_frame.shape[:2]
                    _, _, _ = self.predictor.add_new_points_or_box(
                        inference_state=dummy_state,
                        frame_idx=0,
                        obj_id=1,
                        box=np.array([w//4, h//4, 3*w//4, 3*h//4]),
                    )

                    # Propagate through frames
                    for _ in self.predictor.propagate_in_video(dummy_state):
                        pass

                if self.verbose:
                    print("Warmup complete!")

            except Exception as e:
                if self.verbose:
                    print(f"Warning: Warmup failed with error: {e}")

    def initialize(self, first_frame: np.ndarray) -> bool:
        """
        Initialize tracking with the first frame.

        Args:
            first_frame: First video frame (numpy array, RGB or BGR)

        Returns:
            True if initialization successful
        """
        try:
            # Reset frame buffer and add first frame
            self.frame_buffer = [first_frame]

            with torch.inference_mode():
                # Initialize state with first frame
                self.inference_state = self.predictor.init_state(
                    video_path=self.frame_buffer,
                    offload_video_to_cpu=False,
                    offload_state_to_cpu=False,
                    async_loading_frames=False
                )

            self.is_initialized = True
            self.current_frame_idx = 0

            if self.verbose:
                print("Tracker initialized with first frame")

            return True

        except Exception as e:
            if self.verbose:
                print(f"Error initializing tracker: {e}")
            return False

    def add_object(
        self,
        frame_idx: Optional[int] = None,
        obj_id: Optional[int] = None,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        bbox: Optional[Union[List[int], Tuple[int, int, int, int]]] = None,
        mask: Optional[np.ndarray] = None
    ) -> int:
        """
        Add a new object to track with prompts.

        Args:
            frame_idx: Frame index (None = current frame)
            obj_id: Object ID (None = auto-assign)
            points: Point prompts as (N, 2) array
            labels: Point labels (1=foreground, 0=background)
            bbox: Bounding box as [x1, y1, x2, y2]
            mask: Mask prompt as binary array

        Returns:
            The object ID assigned
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")

        # Auto-assign object ID if not provided
        if obj_id is None:
            obj_id = self.next_obj_id
            self.next_obj_id += 1

        # Use current frame index if not specified
        if frame_idx is None:
            frame_idx = self.current_frame_idx

        # Prepare prompt arguments
        prompt_kwargs = {"frame_idx": frame_idx, "obj_id": obj_id}

        if points is not None:
            prompt_kwargs["points"] = points
            if labels is not None:
                prompt_kwargs["labels"] = labels
            else:
                # Default to foreground points
                prompt_kwargs["labels"] = np.ones(len(points), dtype=np.int32)

        if bbox is not None:
            prompt_kwargs["bbox"] = np.array(bbox)

        if mask is not None:
            prompt_kwargs["mask"] = mask

        # Add prompt to tracker using video predictor API
        try:
            with torch.inference_mode():
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points if points is not None else None,
                    labels=prompt_kwargs.get('labels') if points is not None else None,
                    box=bbox,
                )

            self.object_ids.add(obj_id)

            if self.verbose:
                prompt_type = "bbox" if bbox is not None else "points" if points is not None else "mask"
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
        Track objects in the next frame.

        Args:
            frame: Next video frame (numpy array, RGB or BGR)

        Returns:
            Tuple of (frame_idx, object_ids, masks_dict)
            - frame_idx: Current frame index
            - object_ids: List of tracked object IDs
            - masks_dict: Dictionary mapping object_id -> binary mask
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")

        try:
            # Add new frame to buffer
            self.frame_buffer.append(frame)
            self.current_frame_idx = len(self.frame_buffer) - 1

            # Reinitialize state with updated frame buffer
            # Note: For better performance in long videos, you may want to
            # limit the buffer size to a sliding window (e.g., last 30 frames)
            if len(self.frame_buffer) > 50:  # Sliding window of 50 frames
                # Keep only recent frames
                frames_to_keep = 30
                self.frame_buffer = self.frame_buffer[-frames_to_keep:]

                # Reinitialize state with recent frames
                with torch.inference_mode():
                    self.inference_state = self.predictor.init_state(
                        video_path=self.frame_buffer,
                        offload_video_to_cpu=False,
                        offload_state_to_cpu=False,
                        async_loading_frames=False
                    )

                    # Re-add all tracked objects to the new state
                    for obj_id in list(self.object_ids):
                        # Objects will need to be re-prompted after state reset
                        # This is a limitation of the current approach
                        pass

                self.current_frame_idx = len(self.frame_buffer) - 1

            # Propagate masks through video
            masks_dict = {}
            with torch.inference_mode():
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                    self.inference_state,
                    start_frame_idx=max(0, self.current_frame_idx - 1),
                    max_frame_num_to_track=self.current_frame_idx + 1
                ):
                    if out_frame_idx == self.current_frame_idx:
                        # Convert mask logits to binary masks for current frame
                        for i, obj_id in enumerate(out_obj_ids):
                            mask_logits = out_mask_logits[i]
                            # Threshold mask logits (>0 = foreground)
                            mask = (mask_logits > 0.0).cpu().numpy().squeeze()
                            masks_dict[obj_id] = mask
                        break

            return self.current_frame_idx, list(self.object_ids), masks_dict

        except Exception as e:
            if self.verbose:
                print(f"Error tracking frame {self.current_frame_idx}: {e}")
            raise

    def reset(self):
        """Reset the tracker state."""
        self.inference_state = None
        self.frame_buffer = []
        self.is_initialized = False
        self.current_frame_idx = 0
        self.object_ids = set()

        if self.verbose:
            print("Tracker reset")

    def get_tracked_objects(self) -> List[int]:
        """Get list of currently tracked object IDs."""
        return sorted(list(self.object_ids))

    def remove_object(self, obj_id: int):
        """
        Remove an object from tracking.

        Note: SAM2 may not support dynamic object removal.
        This method updates the internal object list only.
        """
        if obj_id in self.object_ids:
            self.object_ids.remove(obj_id)
            if self.verbose:
                print(f"Removed object {obj_id} from tracking list")
        else:
            if self.verbose:
                print(f"Warning: Object {obj_id} not found in tracking list")

    def __repr__(self) -> str:
        status = "initialized" if self.is_initialized else "not initialized"
        compiled = "compiled" if self.is_compiled else "not compiled"
        return (
            f"SAM2CameraTracker(device={self.device}, "
            f"{status}, {compiled}, "
            f"tracking {len(self.object_ids)} objects)"
        )
