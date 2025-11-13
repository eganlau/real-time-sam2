"""
Real-Time SAM2 - Real-time object tracking with Segment Anything Model 2

This package provides tools for real-time object segmentation and tracking
using Meta's SAM2 model with webcam and video file support.
"""

from .camera_tracker import SAM2CameraTracker
from .input_handler import InputHandler
from .visualizer import Visualizer
from .utils import (
    get_device,
    preprocess_frame,
    calculate_fps,
    Colors,
    FPSCalculator
)

__version__ = "0.1.0"
__author__ = "Real-Time SAM2 Contributors"

__all__ = [
    "SAM2CameraTracker",
    "InputHandler",
    "Visualizer",
    "get_device",
    "preprocess_frame",
    "calculate_fps",
    "Colors",
    "FPSCalculator"
]
