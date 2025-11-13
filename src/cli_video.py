#!/usr/bin/env python3
"""
Video File Processing CLI

Batch process video files with SAM2 tracking and save annotated output.
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import yaml
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from realtime_sam2 import (
    SAM2CameraTracker,
    InputHandler,
    Visualizer,
    FPSCalculator
)


class VideoProcessor:
    """Process video files with SAM2 tracking."""

    def __init__(self, config: dict):
        self.config = config
        self.tracker = None
        self.visualizer = None

    def process_video(
        self,
        input_path: str,
        output_path: str,
        initial_prompts: list = None,
        auto_detect: bool = False
    ):
        """
        Process a video file with SAM2 tracking.

        Args:
            input_path: Path to input video file
            output_path: Path to save output video
            initial_prompts: List of initial prompts (bboxes or points)
            auto_detect: Use automatic detection for initial prompts
        """
        model_config = self.config.get('model', {})
        viz_config = self.config.get('visualization', {})

        print(f"Processing video: {input_path}")
        print(f"Output will be saved to: {output_path}")

        # Initialize tracker
        print("Loading SAM2 model...")
        self.tracker = SAM2CameraTracker(
            config_file=model_config.get('config'),
            checkpoint_path=model_config.get('checkpoint'),
            device=model_config.get('device'),
            use_compile=model_config.get('compile', True),
            use_bfloat16=model_config.get('use_bfloat16', True),
            verbose=True
        )

        # Open input video
        print("Opening input video...")
        input_handler = InputHandler(
            source=input_path,
            verbose=True
        )

        video_info = input_handler.get_frame_info()
        total_frames = video_info['total_frames']
        fps = video_info['fps']
        width = video_info['width']
        height = video_info['height']

        # Initialize visualizer
        self.visualizer = Visualizer(
            alpha=viz_config.get('overlay_alpha', 0.5),
            show_bbox=viz_config.get('show_bbox', True),
            show_labels=viz_config.get('show_labels', True),
            show_fps=False  # Don't show FPS for video processing
        )

        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        if not output_writer.isOpened():
            raise RuntimeError(f"Could not open output video: {output_path}")

        # Read first frame
        ret, first_frame = input_handler.read()
        if not ret:
            raise RuntimeError("Could not read first frame")

        # Warmup if compiled
        if self.tracker.is_compiled:
            print("Warming up model...")
            self.tracker.warmup(first_frame, num_frames=3)

        # Initialize tracker
        print("Initializing tracker...")
        self.tracker.initialize(first_frame)

        # Add initial prompts
        if auto_detect:
            print("Auto-detecting objects in first frame...")
            self._auto_detect_and_add(first_frame)
        elif initial_prompts:
            print(f"Adding {len(initial_prompts)} initial prompts...")
            for prompt in initial_prompts:
                if 'bbox' in prompt:
                    self.tracker.add_object(bbox=prompt['bbox'])
                elif 'points' in prompt:
                    self.tracker.add_object(points=prompt['points'])
        else:
            print("Warning: No initial prompts provided!")
            print("Video will be processed but no objects will be tracked.")

        print(f"Processing {total_frames} frames...")

        # Process video
        frame_count = 0
        masks_dict = {}

        # Write first frame
        vis_frame = self.visualizer.visualize(first_frame, {})
        output_writer.write(vis_frame)
        frame_count += 1

        # Process remaining frames
        with tqdm(total=total_frames-1, desc="Processing") as pbar:
            while True:
                ret, frame = input_handler.read()
                if not ret:
                    break

                # Track
                if len(self.tracker.object_ids) > 0:
                    try:
                        frame_idx, obj_ids, masks_dict = self.tracker.track(frame)
                    except Exception as e:
                        print(f"Error tracking frame {frame_count}: {e}")
                        masks_dict = {}

                # Visualize
                vis_frame = self.visualizer.visualize(frame, masks_dict)

                # Write frame
                output_writer.write(vis_frame)

                frame_count += 1
                pbar.update(1)

        # Cleanup
        input_handler.release()
        output_writer.release()

        print(f"\nProcessing complete!")
        print(f"Processed {frame_count} frames")
        print(f"Output saved to: {output_path}")

    def _auto_detect_and_add(self, frame: np.ndarray):
        """Auto-detect objects and add to tracker."""
        try:
            from realtime_sam2.detector import ObjectDetector

            detector = ObjectDetector(
                model_name="yolov8n.pt",
                confidence_threshold=0.5,
                target_classes=["cell phone"],
                device=str(self.tracker.device),
                verbose=True
            )

            detections = detector.detect(frame, return_format="bbox")

            for det in detections:
                bbox = det['bbox']
                obj_id = self.tracker.add_object(bbox=bbox)
                print(f"  Detected {det['class_name']} with confidence {det['confidence']:.2f}")

            if not detections:
                print("  No objects detected!")

        except Exception as e:
            print(f"Error in auto-detection: {e}")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        return {
            'model': {
                'config': 'sam2/configs/sam2.1/sam2.1_hiera_t.yaml',
                'checkpoint': 'sam2/checkpoints/sam2.1_hiera_tiny.pt',
                'device': None,
                'compile': True,
                'use_bfloat16': True
            },
            'visualization': {
                'overlay_alpha': 0.5,
                'show_bbox': True,
                'show_labels': True
            }
        }


def parse_bbox(bbox_str: str) -> list:
    """Parse bbox string like 'x1,y1,x2,y2' to list."""
    try:
        return [int(x.strip()) for x in bbox_str.split(',')]
    except:
        raise ValueError(f"Invalid bbox format: {bbox_str}. Expected: x1,y1,x2,y2")


def main():
    parser = argparse.ArgumentParser(
        description="Process video files with SAM2 tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect phones in video
  python cli_video.py input.mp4 output.mp4 --auto-detect

  # Track specific object with bbox
  python cli_video.py input.mp4 output.mp4 --bbox 100,100,200,200

  # Track multiple objects
  python cli_video.py input.mp4 output.mp4 --bbox 100,100,200,200 --bbox 300,150,400,250

  # Use custom config
  python cli_video.py input.mp4 output.mp4 --auto-detect --config my_config.yaml
        """
    )

    parser.add_argument(
        'input',
        type=str,
        help='Input video file path'
    )
    parser.add_argument(
        'output',
        type=str,
        help='Output video file path'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--bbox',
        type=str,
        action='append',
        help='Bounding box prompt as x1,y1,x2,y2 (can specify multiple)'
    )
    parser.add_argument(
        '--auto-detect',
        action='store_true',
        help='Automatically detect objects in first frame'
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

    args = parser.parse_args()

    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(args.config)

    # Override with command-line args
    if args.device:
        config['model']['device'] = args.device
    if args.no_compile:
        config['model']['compile'] = False

    # Parse prompts
    initial_prompts = []
    if args.bbox:
        for bbox_str in args.bbox:
            bbox = parse_bbox(bbox_str)
            initial_prompts.append({'bbox': bbox})

    # Process video
    processor = VideoProcessor(config)
    processor.process_video(
        str(input_path),
        str(output_path),
        initial_prompts=initial_prompts if initial_prompts else None,
        auto_detect=args.auto_detect
    )


if __name__ == "__main__":
    main()
