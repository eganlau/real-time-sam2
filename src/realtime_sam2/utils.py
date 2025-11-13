"""
Utility functions and helpers for real-time SAM2 tracking.
"""

import time
from typing import Tuple, Optional
import numpy as np
import torch
import cv2


class Colors:
    """Predefined colors for multi-object visualization (BGR format for OpenCV)."""
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    CYAN = (255, 255, 0)
    ORANGE = (0, 165, 255)
    PURPLE = (128, 0, 128)
    PINK = (203, 192, 255)
    LIME = (0, 255, 128)

    @classmethod
    def get_color(cls, obj_id: int) -> Tuple[int, int, int]:
        """Get a color for an object ID."""
        colors = [
            cls.RED, cls.GREEN, cls.BLUE, cls.YELLOW, cls.MAGENTA,
            cls.CYAN, cls.ORANGE, cls.PURPLE, cls.PINK, cls.LIME
        ]
        return colors[obj_id % len(colors)]

    @classmethod
    def to_rgb(cls, bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Convert BGR to RGB."""
        return (bgr[2], bgr[1], bgr[0])


class FPSCalculator:
    """Calculate frames per second with smoothing."""

    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()

    def update(self) -> float:
        """Update and return current FPS."""
        current_time = time.time()
        delta = current_time - self.last_time
        self.last_time = current_time

        self.frame_times.append(delta)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)

        if len(self.frame_times) > 0:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_time if avg_time > 0 else 0.0
        return 0.0

    def reset(self):
        """Reset the FPS calculator."""
        self.frame_times = []
        self.last_time = time.time()


def get_device(prefer_mps: bool = True, verbose: bool = True) -> torch.device:
    """
    Get the best available device for inference.

    Args:
        prefer_mps: Prefer MPS (Apple Silicon) if available
        verbose: Print device selection info

    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        if verbose:
            print(f"Using CUDA device: {device_name}")
    elif prefer_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("Using MPS (Apple Silicon GPU) device")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Using CPU device (may be slow)")

    return device


def preprocess_frame(
    frame: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Preprocess a frame for SAM2 input.

    Args:
        frame: Input frame (BGR format)
        target_size: Optional (width, height) to resize to

    Returns:
        Preprocessed frame
    """
    if target_size is not None:
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

    return frame


def calculate_fps(start_time: float, frame_count: int) -> float:
    """
    Calculate average FPS.

    Args:
        start_time: Start time in seconds
        frame_count: Number of frames processed

    Returns:
        Average FPS
    """
    elapsed = time.time() - start_time
    if elapsed > 0:
        return frame_count / elapsed
    return 0.0


def normalize_bbox(
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Normalize bounding box coordinates to be within frame bounds.

    Args:
        bbox: (x1, y1, x2, y2) bounding box
        frame_shape: (height, width) of the frame

    Returns:
        Normalized (x1, y1, x2, y2) bounding box
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    return (x1, y1, x2, y2)


def points_to_bbox(points: np.ndarray, padding: int = 10) -> Tuple[int, int, int, int]:
    """
    Convert a set of points to a bounding box.

    Args:
        points: Array of (x, y) points
        padding: Padding to add around the bbox

    Returns:
        (x1, y1, x2, y2) bounding box
    """
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    x1 = int(np.min(x_coords)) - padding
    y1 = int(np.min(y_coords)) - padding
    x2 = int(np.max(x_coords)) + padding
    y2 = int(np.max(y_coords)) + padding

    return (x1, y1, x2, y2)


def mask_to_bbox(mask: np.ndarray, padding: int = 5) -> Optional[Tuple[int, int, int, int]]:
    """
    Convert a binary mask to a bounding box.

    Args:
        mask: Binary mask (H, W)
        padding: Padding to add around the bbox

    Returns:
        (x1, y1, x2, y2) bounding box or None if mask is empty
    """
    if not mask.any():
        return None

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(mask.shape[1] - 1, x2 + padding)
    y2 = min(mask.shape[0] - 1, y2 + padding)

    return (int(x1), int(y1), int(x2), int(y2))


def warmup_model(
    predictor,
    dummy_frame: np.ndarray,
    num_warmup_frames: int = 5,
    verbose: bool = True
):
    """
    Warm up the model with dummy frames (for torch.compile).

    Args:
        predictor: SAM2 predictor instance
        dummy_frame: A dummy frame to use for warmup
        num_warmup_frames: Number of warmup iterations
        verbose: Print warmup progress
    """
    if verbose:
        print(f"Warming up model with {num_warmup_frames} frames...")

    with torch.inference_mode():
        try:
            predictor.load_first_frame(dummy_frame)

            for i in range(num_warmup_frames):
                predictor.track(dummy_frame)
                if verbose:
                    print(f"  Warmup frame {i+1}/{num_warmup_frames}")
        except Exception as e:
            if verbose:
                print(f"Warning: Warmup failed with error: {e}")

    # Reset predictor state after warmup
    try:
        predictor.reset_state()
    except:
        pass

    if verbose:
        print("Warmup complete!")
