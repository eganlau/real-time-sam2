# SAM2 Tracker Modes Documentation

This document explains the two tracking modes available in real-time-sam2.

## Tracker Modes

This project supports **two different tracking backends**, each optimized for different use cases:

### Camera Mode (Default) - SAM2CameraPredictor

Based on Meta's official `segment-anything-2-real-time` implementation.

**Features:**
- ✅ **Manual object selection** via click-and-drag bounding boxes
- ✅ **FIFO memory management** (keeps last 7 frames)
- ✅ **Interactive tracking** with user control
- ✅ **Stable performance** for controlled scenes
- ✅ **Best for**: Interactive demos, controlled environments, user-guided tracking

**Usage:**
```bash
# Default mode
python src/cli_webcam.py --device mps

# Explicit camera mode
python src/cli_webcam.py --device mps --tracker-mode camera
```

**How it works:**
- User selects objects by clicking and dragging bounding boxes
- SAM2 segments the selected region
- FIFO memory buffer maintains temporal consistency across last 7 frames
- Automatic memory pruning keeps GPU memory usage low

### Kalman Mode - SAM2ObjectTracker (SAMURAI)

Based on the `sam2_realtime` (SAMURAI) implementation with Kalman filter integration.

**Features:**
- ✅ **Autonomous tracking** with motion prediction
- ✅ **Dual memory banks** (short-term 7 + long-term 7 frames)
- ✅ **Kalman filter** for bbox prediction and smoothing
- ✅ **Occlusion handling** via memory promotion
- ⚠️ **Manual selection NOT supported** - uses automatic detection
- ✅ **Best for**: Autonomous systems, moving cameras, occlusions

**Usage:**
```bash
python src/cli_webcam.py --device mps --tracker-mode kalman
```

**How it works:**
- Automatically detects and tracks objects (no manual selection)
- Kalman filter predicts object motion between frames
- Short-term memory: Last 7 frames for recent tracking
- Long-term memory: Promoted frames with visible objects (object_score > 5)
- Weighted IoU fusion: `0.15 * kf_iou + 0.85 * sam_iou`

### Comparison

| Feature | Camera Mode | Kalman Mode |
|---------|-------------|-------------|
| **Manual Selection** | ✅ Yes | ❌ No (automatic only) |
| **Memory Strategy** | FIFO (7 frames) | Dual banks (7+7) |
| **Motion Prediction** | ❌ No | ✅ Kalman filter |
| **Occlusion Handling** | Basic | Advanced (promotion) |
| **Use Case** | Interactive demos | Autonomous tracking |
| **Complexity** | Simple | Complex |
| **Dependencies** | SAM2 only | SAM2 + filterpy |

### Which Mode Should I Use?

**Choose Camera Mode if:**
- You want to manually select objects to track
- You're building an interactive demo or annotation tool
- Your scene is relatively stable
- You prefer simpler, more predictable behavior

**Choose Kalman Mode if:**
- You need fully autonomous tracking
- Your camera or objects move significantly
- You have occlusions or partial visibility
- You want motion prediction and smoothing
- You're willing to trade control for robustness

### Configuration

You can set the default tracker mode in `configs/config.yaml`:

```yaml
tracking:
  tracker_mode: "camera"  # or "kalman"
```

Or override via command line:

```bash
python src/cli_webcam.py --tracker-mode kalman
```

## Technical Details

### Camera Mode Architecture

```
Frame → Image Encoding → Memory Attention → Mask Decoder → Masks
                ↑              ↑
                └──────────────┴──── FIFO Buffer (7 frames)
```

### Kalman Mode Architecture

```
Frame → Image Encoding → Kalman Prediction → Memory Attention → Mask Decoder → Masks
                ↑              ↑                    ↑
                └──────────────┼────────────────────┘
                              Short-term (7) + Long-term (7) Memory Banks
```

## References

- **Camera Mode**: [segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time)
- **Kalman Mode**: [SAMURAI (sam2_realtime)](https://github.com/yangchris11/samurai)
- **SAM2**: [Meta's Segment Anything Model 2](https://github.com/facebookresearch/sam2)
