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
            from sam2.build_sam import build_sam2_camera_predictor
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
        self.predictor = build_sam2_camera_predictor(
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=str(self.device)
        )

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

    def warmup(self, dummy_frame: np.ndarray, num_frames: int = 5):
        """
        Warm up the model for torch.compile optimization.

        Args:
            dummy_frame: A representative frame for warmup
            num_frames: Number of warmup iterations
        """
        if self.is_compiled:
            warmup_model(
                self.predictor,
                dummy_frame,
                num_frames,
                verbose=self.verbose
            )

    def initialize(self, first_frame: np.ndarray) -> bool:
        """
        Initialize tracking with the first frame.

        Args:
            first_frame: First video frame (numpy array, RGB or BGR)

        Returns:
            True if initialization successful
        """
        try:
            with torch.inference_mode():
                if self.use_bfloat16 and self.device.type in ["cuda", "mps"]:
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        self.predictor.load_first_frame(first_frame)
                else:
                    self.predictor.load_first_frame(first_frame)

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

        # Add prompt to tracker
        try:
            with torch.inference_mode():
                if self.use_bfloat16 and self.device.type in ["cuda", "mps"]:
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        self.predictor.add_new_prompt(**prompt_kwargs)
                else:
                    self.predictor.add_new_prompt(**prompt_kwargs)

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
            with torch.inference_mode():
                if self.use_bfloat16 and self.device.type in ["cuda", "mps"]:
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        out_frame_idx, out_obj_ids, out_mask_logits = self.predictor.track(frame)
                else:
                    out_frame_idx, out_obj_ids, out_mask_logits = self.predictor.track(frame)

            self.current_frame_idx = out_frame_idx

            # Convert mask logits to binary masks
            masks_dict = {}
            for obj_id, mask_logits in zip(out_obj_ids, out_mask_logits):
                # Threshold mask logits (>0 = foreground)
                mask = (mask_logits > 0.0).cpu().numpy().squeeze()
                masks_dict[obj_id] = mask

            return out_frame_idx, list(out_obj_ids), masks_dict

        except Exception as e:
            if self.verbose:
                print(f"Error tracking frame {self.current_frame_idx}: {e}")
            raise

    def reset(self):
        """Reset the tracker state."""
        try:
            self.predictor.reset_state()
        except AttributeError:
            # reset_state may not be available, create new predictor
            if self.verbose:
                print("Warning: reset_state not available, tracker state may persist")

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
