"""
Input Handler - Webcam and video file management for real-time tracking.

This module provides a unified interface for capturing frames from
webcams or video files with camera auto-detection support.
"""

import cv2
import platform
from typing import Optional, Tuple
import numpy as np


class InputHandler:
    """
    Unified input handler for webcam and video file sources.

    Supports automatic external webcam detection and provides a
    consistent interface for frame capture.
    """

    def __init__(
        self,
        source: Optional[int | str] = None,
        prefer_external: bool = True,
        target_fps: Optional[int] = None,
        target_resolution: Optional[Tuple[int, int]] = None,
        verbose: bool = True
    ):
        """
        Initialize the input handler.

        Args:
            source: Video source (int for webcam, str for video file, None for auto-detect)
            prefer_external: Prefer external webcam over built-in
            target_fps: Target capture FPS (None = use camera default)
            target_resolution: Target resolution as (width, height)
            verbose: Print status messages
        """
        self.verbose = verbose
        self.source = source
        self.prefer_external = prefer_external
        self.target_fps = target_fps
        self.target_resolution = target_resolution
        self.cap = None
        self.is_webcam = False
        self.total_frames = None
        self.fps = None

        self._open_source()

    def _detect_cameras(self) -> list:
        """
        Detect available cameras on the system.

        Returns:
            List of available camera indices
        """
        available = []
        max_test = 10  # Test first 10 indices

        for idx in range(max_test):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                # Test if we can actually read a frame
                ret, _ = cap.read()
                if ret:
                    available.append(idx)
            cap.release()

        return available

    def _get_best_camera(self) -> int:
        """
        Get the best camera index based on preferences.

        For macOS, external cameras typically have higher indices.
        For other systems, we assume index 0 is built-in and higher are external.

        Returns:
            Camera index to use
        """
        available = self._detect_cameras()

        if not available:
            if self.verbose:
                print("Warning: No cameras detected, defaulting to index 0")
            return 0

        if self.verbose:
            print(f"Detected cameras at indices: {available}")

        if self.prefer_external and len(available) > 1:
            # Prefer highest index (likely external)
            camera_idx = available[-1]
            if self.verbose:
                print(f"Using external camera at index {camera_idx}")
            return camera_idx
        else:
            # Use first available (likely built-in)
            camera_idx = available[0]
            if self.verbose:
                print(f"Using camera at index {camera_idx}")
            return camera_idx

    def _open_source(self):
        """Open the video source (webcam or file)."""
        if self.source is None:
            # Auto-detect best camera
            self.source = self._get_best_camera()
            self.is_webcam = True
        elif isinstance(self.source, int):
            # Explicit camera index
            self.is_webcam = True
        else:
            # Video file path
            self.is_webcam = False
            if self.verbose:
                print(f"Opening video file: {self.source}")

        # Open capture
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not self.is_webcam:
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self.total_frames = None

        # Set target FPS if specified
        if self.target_fps is not None and self.is_webcam:
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # Set target resolution if specified
        if self.target_resolution is not None and self.is_webcam:
            target_w, target_h = self.target_resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)

            # Verify the resolution was set
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.verbose:
            source_type = "webcam" if self.is_webcam else "video file"
            print(f"Opened {source_type}: {self.width}x{self.height} @ {self.fps:.1f} FPS")
            if self.total_frames:
                print(f"Total frames: {self.total_frames}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame.

        Returns:
            Tuple of (success, frame)
            - success: True if frame was read successfully
            - frame: Frame as numpy array (BGR format) or None
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        return ret, frame

    def get_frame_info(self) -> dict:
        """Get information about the current frame/video."""
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "is_webcam": self.is_webcam,
            "source": self.source
        }

    def get_current_frame_number(self) -> Optional[int]:
        """Get current frame number (None for webcam)."""
        if self.is_webcam:
            return None
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def get_progress(self) -> Optional[float]:
        """
        Get playback progress as percentage (0-100).

        Returns:
            Progress percentage or None for webcam
        """
        if self.is_webcam or self.total_frames is None:
            return None

        current = self.get_current_frame_number()
        if current is not None and self.total_frames > 0:
            return (current / self.total_frames) * 100.0
        return None

    def seek(self, frame_number: int) -> bool:
        """
        Seek to a specific frame (video files only).

        Args:
            frame_number: Frame number to seek to

        Returns:
            True if seek was successful
        """
        if self.is_webcam:
            if self.verbose:
                print("Warning: Cannot seek in webcam mode")
            return False

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        return True

    def restart(self) -> bool:
        """
        Restart video from beginning (video files only).

        Returns:
            True if restart was successful
        """
        if self.is_webcam:
            return False

        return self.seek(0)

    def release(self):
        """Release the video capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            if self.verbose:
                print("Input handler released")

    def is_opened(self) -> bool:
        """Check if video source is opened."""
        return self.cap is not None and self.cap.isOpened()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

    def __repr__(self) -> str:
        source_type = "webcam" if self.is_webcam else "video"
        status = "opened" if self.is_opened() else "closed"
        return (
            f"InputHandler({source_type}, "
            f"{self.width}x{self.height}, "
            f"{self.fps:.1f} FPS, {status})"
        )
