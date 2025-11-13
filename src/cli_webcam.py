#!/usr/bin/env python3
"""
Real-Time Webcam Tracking CLI

Interactive real-time object tracking using SAM2 with webcam input.
Supports manual object selection and optional automatic detection.
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from realtime_sam2 import (
    InputHandler,
    Visualizer,
    FPSCalculator,
    Colors
)
from realtime_sam2.camera_tracker import SAM2CameraTracker
from realtime_sam2.kalman_tracker import SAM2KalmanTracker


class InteractiveTracker:
    """Interactive webcam tracker with mouse selection support."""

    def __init__(self, config: dict):
        self.config = config
        self.tracker = None
        self.input_handler = None
        self.visualizer = None
        self.fps_calculator = FPSCalculator()

        # State
        self.is_paused = False
        self.is_selecting = False
        self.selection_start = None
        self.selection_end = None
        self.current_frame = None
        self.masks_dict = {}

        # Auto-detection
        self.detector = None
        self.use_auto_detect = config.get('tracking', {}).get('mode') in ['auto', 'both']
        self.auto_detect_interval = config.get('tracking', {}).get('auto_detect_interval', 30)
        self.frame_count = 0

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for object selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selection_start = (x, y)
            self.is_selecting = True
        elif event == cv2.EVENT_MOUSEMOVE and self.is_selecting:
            self.selection_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.selection_end = (x, y)
            self.is_selecting = False
            self.add_selected_object()

    def add_selected_object(self):
        """Add selected region to tracker."""
        if self.selection_start and self.selection_end and self.current_frame is not None:
            x1, y1 = self.selection_start
            x2, y2 = self.selection_end

            # Normalize coordinates
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Add to tracker with current frame
            try:
                obj_id = self.tracker.add_object(
                    frame=self.current_frame,
                    bbox=[x1, y1, x2, y2]
                )
                print(f"Added object {obj_id} with bbox [{x1}, {y1}, {x2}, {y2}]")
            except Exception as e:
                print(f"Error adding object: {e}")

            # Reset selection
            self.selection_start = None
            self.selection_end = None

    def auto_detect_objects(self, frame: np.ndarray):
        """Automatically detect and track objects."""
        if self.detector is None:
            try:
                from realtime_sam2.detector import ObjectDetector
                self.detector = ObjectDetector(
                    model_name="yolov8n.pt",
                    confidence_threshold=0.5,
                    target_classes=["cell phone"],
                    device=str(self.tracker.device),
                    verbose=False
                )
            except Exception as e:
                print(f"Warning: Could not load detector: {e}")
                self.use_auto_detect = False
                return

        # Detect phones
        try:
            detections = self.detector.detect(frame, return_format="bbox")

            for det in detections:
                # Add detected object to tracker
                bbox = det['bbox']
                obj_id = self.tracker.add_object(frame=frame, bbox=bbox)
                print(f"Auto-detected {det['class_name']} (confidence: {det['confidence']:.2f})")

        except Exception as e:
            print(f"Error in auto-detection: {e}")

    def run(self):
        """Run the interactive tracker."""
        # Initialize components
        model_config = self.config.get('model', {})
        input_config = self.config.get('input', {})
        viz_config = self.config.get('visualization', {})
        tracking_config = self.config.get('tracking', {})

        # Get tracker mode
        tracker_mode = tracking_config.get('tracker_mode', 'camera')

        print(f"Initializing SAM2 Tracker (mode: {tracker_mode})...")

        # Factory pattern: create appropriate tracker
        if tracker_mode == 'camera':
            self.tracker = SAM2CameraTracker(
                config_file=model_config.get('config', 'configs/sam2.1/sam2.1_hiera_t.yaml'),
                checkpoint_path=model_config.get('checkpoint', 'sam2/checkpoints/sam2.1_hiera_tiny.pt'),
                device=model_config.get('device'),
                verbose=True
            )
        elif tracker_mode == 'kalman':
            self.tracker = SAM2KalmanTracker(
                config_file=model_config.get('config', 'configs/sam2.1/sam2.1_hiera_t.yaml'),
                checkpoint_path=model_config.get('checkpoint', 'sam2/checkpoints/sam2.1_hiera_tiny.pt'),
                device=model_config.get('device'),
                num_objects=tracking_config.get('max_objects', 10),
                verbose=True
            )
            print("NOTE: Kalman tracker uses automatic detection, not manual bbox selection")
        else:
            raise ValueError(f"Unknown tracker mode: {tracker_mode}. Choose 'camera' or 'kalman'")

        # Open webcam
        print("Opening webcam...")
        self.input_handler = InputHandler(
            source=input_config.get('source'),
            prefer_external=input_config.get('prefer_external_webcam', True),
            target_resolution=tuple(input_config.get('resolution', [640, 480])),
            verbose=True
        )

        # Initialize visualizer
        self.visualizer = Visualizer(
            alpha=viz_config.get('overlay_alpha', 0.5),
            show_bbox=viz_config.get('show_bbox', True),
            show_labels=viz_config.get('show_labels', True),
            show_fps=viz_config.get('show_fps', True)
        )

        # Read first frame
        ret, first_frame = self.input_handler.read()
        if not ret:
            print("Error: Could not read from webcam")
            return

        # Initialize tracker
        print("Initializing tracker...")
        self.tracker.initialize(first_frame)
        self.current_frame = first_frame

        # Setup window
        window_name = "SAM2 Real-Time Tracker"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        # Instructions
        instructions = [
            "Controls:",
            "  [Space] - Pause/Resume",
            "  [R] - Reset tracking",
            "  [A] - Auto-detect (if enabled)",
            "  [Q] - Quit",
            "  [Mouse] - Click and drag to select object"
        ]

        print("\n" + "\n".join(instructions))
        print("\nStarting real-time tracking...")

        # Main loop
        try:
            while True:
                if not self.is_paused:
                    # Read frame
                    ret, frame = self.input_handler.read()
                    if not ret:
                        print("Error reading frame")
                        break

                    self.current_frame = frame
                    self.frame_count += 1

                    # Auto-detect on interval (only if tracking mode allows it)
                    if (self.use_auto_detect and
                        self.frame_count % self.auto_detect_interval == 0 and
                        len(self.tracker.tracked_objects) < self.config.get('tracking', {}).get('max_objects', 10) and
                        self.tracker.is_initialized):
                        self.auto_detect_objects(frame)

                    # Track objects
                    if len(self.tracker.tracked_objects) > 0:
                        frame_idx, obj_ids, self.masks_dict = self.tracker.track(frame)
                    else:
                        self.masks_dict = {}

                    # Update FPS
                    fps = self.fps_calculator.update()

                    # Visualize
                    vis_frame = self.visualizer.visualize(
                        frame,
                        self.masks_dict,
                        fps=fps,
                        instructions=instructions if len(self.tracker.tracked_objects) == 0 else None
                    )
                else:
                    # Paused - use current frame
                    vis_frame = self.current_frame.copy()
                    fps = 0

                # Draw selection box if selecting
                if self.is_selecting and self.selection_start and self.selection_end:
                    cv2.rectangle(
                        vis_frame,
                        self.selection_start,
                        self.selection_end,
                        (0, 255, 0),
                        2
                    )

                # Show frame
                cv2.imshow(window_name, vis_frame)

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord(' '):
                    self.is_paused = not self.is_paused
                    status = "Paused" if self.is_paused else "Resumed"
                    print(f"{status}")
                elif key == ord('r'):
                    print("Resetting tracker...")
                    self.tracker.reset()
                    self.tracker.initialize(self.current_frame)
                    self.masks_dict = {}
                    self.fps_calculator.reset()
                    print("Tracker reset")
                elif key == ord('a') and self.use_auto_detect:
                    print("Running auto-detection...")
                    self.auto_detect_objects(self.current_frame)

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            # Cleanup
            print("Cleaning up...")
            self.input_handler.release()
            cv2.destroyAllWindows()
            print("Done!")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        # Return default config
        return {
            'model': {
                'config': 'sam2/configs/sam2.1/sam2.1_hiera_t.yaml',
                'checkpoint': 'sam2/checkpoints/sam2.1_hiera_tiny.pt',
                'device': None,
                'compile': True,
                'use_bfloat16': True
            },
            'input': {
                'prefer_external_webcam': True,
                'resolution': [640, 480]
            },
            'tracking': {
                'mode': 'both',
                'max_objects': 10,
                'auto_detect_interval': 30
            },
            'visualization': {
                'overlay_alpha': 0.5,
                'show_bbox': True,
                'show_labels': True,
                'show_fps': True
            }
        }


def main():
    parser = argparse.ArgumentParser(
        description="Real-time object tracking with SAM2 and webcam"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use (overrides config)'
    )
    parser.add_argument(
        '--no-compile',
        action='store_true',
        help='Disable torch.compile'
    )
    parser.add_argument(
        '--tracker-mode',
        type=str,
        choices=['camera', 'kalman'],
        default='camera',
        help='Tracker mode: camera (manual selection, FIFO memory) or kalman (automatic, dual memory with Kalman filter)'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with command-line args
    if args.device:
        config['model']['device'] = args.device
    if args.no_compile:
        config['model']['compile'] = False
    if args.tracker_mode:
        if 'tracking' not in config:
            config['tracking'] = {}
        config['tracking']['tracker_mode'] = args.tracker_mode

    # Run tracker
    tracker = InteractiveTracker(config)
    tracker.run()


if __name__ == "__main__":
    main()
