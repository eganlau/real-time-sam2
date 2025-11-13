"""
Visualizer - Real-time visualization of segmentation masks and tracking results.

This module provides tools for overlaying masks, drawing bounding boxes,
and displaying tracking information in real-time.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

from .utils import Colors, mask_to_bbox


class Visualizer:
    """
    Visualize segmentation masks and tracking results on video frames.

    Provides tools for overlaying masks with transparency, drawing bounding
    boxes, and displaying tracking metadata like FPS and object IDs.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        show_bbox: bool = True,
        show_labels: bool = True,
        show_fps: bool = True,
        bbox_thickness: int = 2,
        label_font_scale: float = 0.6,
        label_thickness: int = 2
    ):
        """
        Initialize the visualizer.

        Args:
            alpha: Transparency of mask overlay (0=transparent, 1=opaque)
            show_bbox: Draw bounding boxes around objects
            show_labels: Show object ID labels
            show_fps: Display FPS counter
            bbox_thickness: Thickness of bounding box lines
            label_font_scale: Font scale for labels
            label_thickness: Thickness of label text
        """
        self.alpha = alpha
        self.show_bbox = show_bbox
        self.show_labels = show_labels
        self.show_fps = show_fps
        self.bbox_thickness = bbox_thickness
        self.label_font_scale = label_font_scale
        self.label_thickness = label_thickness

    def overlay_masks(
        self,
        frame: np.ndarray,
        masks_dict: Dict[int, np.ndarray],
        colors: Optional[Dict[int, Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        Overlay segmentation masks on a frame.

        Args:
            frame: Input frame (BGR format)
            masks_dict: Dictionary mapping object_id -> binary mask
            colors: Optional dictionary mapping object_id -> BGR color

        Returns:
            Frame with overlaid masks
        """
        if not masks_dict:
            return frame

        # Create overlay copy
        overlay = frame.copy()

        # Apply each mask
        for obj_id, mask in masks_dict.items():
            if mask is None or not mask.any():
                continue

            # Get color for this object
            if colors and obj_id in colors:
                color = colors[obj_id]
            else:
                color = Colors.get_color(obj_id)

            # Convert tensor to numpy if needed
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()

            # Squeeze any extra dimensions (e.g., (1, H, W) -> (H, W))
            while mask.ndim > 2:
                mask = mask.squeeze()

            # Ensure we have valid 2D mask
            if mask.ndim != 2:
                continue

            # Resize mask if needed
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            # Ensure mask is boolean for indexing
            mask = mask.astype(bool)

            # Apply color to mask region
            overlay[mask] = color

        # Blend overlay with original frame
        result = cv2.addWeighted(frame, 1 - self.alpha, overlay, self.alpha, 0)

        return result

    def draw_bboxes(
        self,
        frame: np.ndarray,
        masks_dict: Dict[int, np.ndarray],
        colors: Optional[Dict[int, Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        Draw bounding boxes around segmented objects.

        Args:
            frame: Input frame (BGR format)
            masks_dict: Dictionary mapping object_id -> binary mask
            colors: Optional dictionary mapping object_id -> BGR color

        Returns:
            Frame with bounding boxes drawn
        """
        if not self.show_bbox or not masks_dict:
            return frame

        result = frame.copy()

        for obj_id, mask in masks_dict.items():
            if mask is None or not mask.any():
                continue

            # Convert tensor to numpy if needed
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()

            # Squeeze any extra dimensions
            while mask.ndim > 2:
                mask = mask.squeeze()

            # Skip invalid masks
            if mask.ndim != 2:
                continue

            # Resize mask if needed
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            # Get bounding box from mask
            bbox = mask_to_bbox(mask, padding=0)
            if bbox is None:
                continue

            # Get color
            if colors and obj_id in colors:
                color = colors[obj_id]
            else:
                color = Colors.get_color(obj_id)

            # Draw rectangle
            x1, y1, x2, y2 = bbox
            cv2.rectangle(result, (x1, y1), (x2, y2), color, self.bbox_thickness)

        return result

    def draw_labels(
        self,
        frame: np.ndarray,
        masks_dict: Dict[int, np.ndarray],
        colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
        labels: Optional[Dict[int, str]] = None
    ) -> np.ndarray:
        """
        Draw object ID labels on segmented objects.

        Args:
            frame: Input frame (BGR format)
            masks_dict: Dictionary mapping object_id -> binary mask
            colors: Optional dictionary mapping object_id -> BGR color
            labels: Optional dictionary mapping object_id -> label text

        Returns:
            Frame with labels drawn
        """
        if not self.show_labels or not masks_dict:
            return frame

        result = frame.copy()

        for obj_id, mask in masks_dict.items():
            if mask is None or not mask.any():
                continue

            # Convert tensor to numpy if needed
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()

            # Squeeze any extra dimensions
            while mask.ndim > 2:
                mask = mask.squeeze()

            # Skip invalid masks
            if mask.ndim != 2:
                continue

            # Resize mask if needed
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            # Get bounding box for label position
            bbox = mask_to_bbox(mask, padding=0)
            if bbox is None:
                continue

            # Get color
            if colors and obj_id in colors:
                color = colors[obj_id]
            else:
                color = Colors.get_color(obj_id)

            # Get label text
            if labels and obj_id in labels:
                label_text = labels[obj_id]
            else:
                label_text = f"ID: {obj_id}"

            # Draw label background and text
            x1, y1, x2, y2 = bbox
            label_size, baseline = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.label_font_scale,
                self.label_thickness
            )

            # Background rectangle
            cv2.rectangle(
                result,
                (x1, y1 - label_size[1] - baseline - 5),
                (x1 + label_size[0], y1),
                color,
                -1  # Filled
            )

            # Text
            cv2.putText(
                result,
                label_text,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.label_font_scale,
                (255, 255, 255),  # White text
                self.label_thickness
            )

        return result

    def draw_fps(
        self,
        frame: np.ndarray,
        fps: float,
        position: str = "top_left"
    ) -> np.ndarray:
        """
        Draw FPS counter on frame.

        Args:
            frame: Input frame (BGR format)
            fps: Current FPS value
            position: Position on frame ('top_left', 'top_right', 'bottom_left', 'bottom_right')

        Returns:
            Frame with FPS counter
        """
        if not self.show_fps:
            return frame

        result = frame.copy()
        fps_text = f"FPS: {fps:.1f}"

        # Get text size
        text_size, baseline = cv2.getTextSize(
            fps_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )

        # Calculate position
        h, w = frame.shape[:2]
        padding = 10

        if position == "top_left":
            x, y = padding, text_size[1] + padding
        elif position == "top_right":
            x, y = w - text_size[0] - padding, text_size[1] + padding
        elif position == "bottom_left":
            x, y = padding, h - padding
        elif position == "bottom_right":
            x, y = w - text_size[0] - padding, h - padding
        else:
            x, y = padding, text_size[1] + padding

        # Draw background
        cv2.rectangle(
            result,
            (x - 5, y - text_size[1] - 5),
            (x + text_size[0] + 5, y + 5),
            (0, 0, 0),
            -1
        )

        # Draw text
        cv2.putText(
            result,
            fps_text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),  # Green
            2
        )

        return result

    def draw_instructions(
        self,
        frame: np.ndarray,
        instructions: List[str],
        position: str = "bottom_left"
    ) -> np.ndarray:
        """
        Draw instruction text on frame.

        Args:
            frame: Input frame (BGR format)
            instructions: List of instruction strings
            position: Position on frame

        Returns:
            Frame with instructions
        """
        if not instructions:
            return frame

        result = frame.copy()
        h, w = frame.shape[:2]
        padding = 10
        line_height = 25

        # Calculate starting position
        if position == "bottom_left":
            x = padding
            y_start = h - padding - (len(instructions) * line_height)
        elif position == "bottom_right":
            # Will adjust x per line based on text width
            y_start = h - padding - (len(instructions) * line_height)
        else:
            x = padding
            y_start = padding + 50  # Below FPS counter

        # Draw each instruction
        for i, instruction in enumerate(instructions):
            y = y_start + (i * line_height)

            if position == "bottom_right":
                # Right-align text
                text_size, _ = cv2.getTextSize(
                    instruction,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    1
                )
                x = w - text_size[0] - padding

            # Draw text with shadow for readability
            # Shadow
            cv2.putText(
                result,
                instruction,
                (x + 1, y + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )
            # Main text
            cv2.putText(
                result,
                instruction,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        return result

    def visualize(
        self,
        frame: np.ndarray,
        masks_dict: Dict[int, np.ndarray],
        fps: Optional[float] = None,
        colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
        labels: Optional[Dict[int, str]] = None,
        instructions: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Complete visualization with all enabled features.

        Args:
            frame: Input frame (BGR format)
            masks_dict: Dictionary mapping object_id -> binary mask
            fps: Optional FPS value to display
            colors: Optional dictionary mapping object_id -> BGR color
            labels: Optional dictionary mapping object_id -> label text
            instructions: Optional instruction text to display

        Returns:
            Fully visualized frame
        """
        # Overlay masks
        result = self.overlay_masks(frame, masks_dict, colors)

        # Draw bounding boxes
        result = self.draw_bboxes(result, masks_dict, colors)

        # Draw labels
        result = self.draw_labels(result, masks_dict, colors, labels)

        # Draw FPS
        if fps is not None:
            result = self.draw_fps(result, fps, position="top_left")

        # Draw instructions
        if instructions:
            result = self.draw_instructions(result, instructions, position="bottom_left")

        return result
