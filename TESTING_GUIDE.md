# Phone Tracking Testing Guide

This guide will help you test both **manual selection** and **automatic phone detection** modes.

---

## Prerequisites

1. **Pull latest changes** on your Mac:
   ```bash
   cd ~/dev/nlp/real-time-sam2
   git pull
   ```

2. **Install YOLO** (needed for automatic detection):
   ```bash
   pip install ultralytics
   ```

---

## Test 1: Manual Phone Selection (Camera Mode)

**What this tests:** Interactive click-and-drag selection with temporal memory.

### Setup

No config changes needed - this is the default mode.

### Run

```bash
python src/cli_webcam.py --device mps --tracker-mode camera
```

### How to Test

1. **Position your phone** in front of the webcam
2. **Click and drag** a bounding box around the phone
3. **Move the phone around** - watch the mask follow it
4. **Test scenarios:**
   - ✅ Slow movements (should track perfectly)
   - ✅ Fast movements (test if it keeps up)
   - ✅ Partial occlusion (cover part of phone with hand)
   - ✅ Rotation (rotate phone 90°, 180°)
   - ✅ Distance changes (move phone closer/farther)

### Controls

- **Mouse**: Click and drag to select phone
- **Space**: Pause/Resume
- **R**: Reset tracking (clears all objects)
- **Q**: Quit

### Expected Results

**Good tracking:**
- ✅ Mask stays on phone during movement
- ✅ Mask updates shape when phone rotates
- ✅ Recovers after brief occlusions

**What might fail:**
- ⚠️ Very fast movements (>30% of frame in one frame)
- ⚠️ Complete occlusion (phone fully hidden)
- ⚠️ Similar objects nearby (might switch targets)

---

## Test 2: Automatic Phone Detection (Auto Mode)

**What this tests:** YOLO detects phones, SAM2 segments them automatically.

### Setup

**Option A: Config file (persistent)**

Edit `configs/config.yaml`:
```yaml
tracking:
  tracker_mode: "camera"
  mode: "auto"  # Change from "manual" to "auto"
  confidence_threshold: 0.5
  target_classes:
    - "cell phone"
```

Then run:
```bash
python src/cli_webcam.py --device mps
```

**Option B: Keep manual as default, test auto temporarily**

Create a test config file:
```bash
cp configs/config.yaml configs/config_auto.yaml
```

Edit `configs/config_auto.yaml`:
```yaml
tracking:
  mode: "auto"
```

Run with test config:
```bash
python src/cli_webcam.py --device mps --config configs/config_auto.yaml
```

### How to Test

1. **Just hold your phone** in front of the camera
2. **Wait ~1-2 seconds** for YOLO to detect it
3. **Watch automatic tracking** - no clicking needed!
4. **Test scenarios:**
   - ✅ Single phone (should detect and track)
   - ✅ Multiple phones (tracks all simultaneously)
   - ✅ Phone appears/disappears (re-detection every 30 frames)
   - ✅ Different phone orientations
   - ✅ Different lighting conditions

### Controls

- **A**: Manually trigger auto-detection now (don't wait for interval)
- **Space**: Pause/Resume
- **R**: Reset tracking
- **Q**: Quit

### Expected Results

**Good detection:**
- ✅ Phone detected within 1-2 seconds
- ✅ Green bbox appears around phone
- ✅ Colored mask overlay on phone
- ✅ Tracks automatically without user input

**What might fail:**
- ⚠️ Phone in unusual orientation (sideways, upside down)
- ⚠️ Phone screen off (harder to detect than screen on)
- ⚠️ Very small phone in frame (<50px)
- ⚠️ False positives (might detect TV remote, tablet, etc.)

---

## Test 3: Hybrid Mode (Manual + Auto)

**What this tests:** Automatic detection + ability to manually add more objects.

### Setup

Edit `configs/config.yaml`:
```yaml
tracking:
  mode: "both"
```

### How to Test

1. **Hold phone** - wait for auto-detection
2. **Manually select another object** (your hand, face, cup, etc.)
3. **Now both are tracked**: phone (auto) + your object (manual)

### Expected Results

- ✅ Phone detected automatically
- ✅ Can still click-and-drag to add more objects
- ✅ All objects tracked simultaneously

---

## Performance Comparison Matrix

Test both modes and fill this out:

| Scenario | Manual Mode | Auto Mode | Winner |
|----------|-------------|-----------|--------|
| **Initial Detection** | Click-drag (1 sec) | YOLO detects (1-2 sec) | ??? |
| **Tracking Accuracy** | ??? | ??? | ??? |
| **Movement Handling** | ??? | ??? | ??? |
| **Occlusion Recovery** | ??? | ??? | ??? |
| **Multi-phone** | Click each one | All detected | ??? |
| **CPU/GPU Usage** | Lower | Higher (YOLO + SAM2) | Manual |
| **FPS** | ??? fps | ??? fps | ??? |
| **Ease of Use** | Requires clicking | Hands-free | Auto |

---

## Troubleshooting

### "YOLO not found" error
```bash
pip install ultralytics
```

### Auto-detection not working
1. Check console for YOLO loading messages
2. Press **A** key to manually trigger detection
3. Lower confidence threshold in config:
   ```yaml
   confidence_threshold: 0.3  # Try lower threshold
   ```

### Phone not detected
- ✅ Make sure phone screen is ON (easier to detect)
- ✅ Hold phone upright (YOLO trained on typical phone orientations)
- ✅ Ensure good lighting
- ✅ Phone should be >10% of frame size

### Too many false positives
Increase confidence:
```yaml
confidence_threshold: 0.7  # Higher = more strict
```

### Performance too slow
1. Use tiny model (already default):
   ```yaml
   model:
     config: "configs/sam2.1/sam2.1_hiera_t.yaml"
     checkpoint: "sam2/checkpoints/sam2.1_hiera_tiny.pt"
   ```

2. Lower webcam resolution:
   ```yaml
   input:
     resolution: [480, 360]  # Lower than 640x480
   ```

3. Skip frames:
   ```yaml
   performance:
     frame_skip: 2  # Process every 2nd frame
   ```

---

## Recommended Testing Order

1. **Start with Manual Mode** (Test 1)
   - Get familiar with the interface
   - Understand what "good tracking" looks like
   - Test edge cases

2. **Try Auto Mode** (Test 2)
   - Compare detection speed vs manual
   - Test multi-phone scenarios
   - Note any false positives

3. **Test Hybrid Mode** (Test 3)
   - Best of both worlds
   - See if auto-detection + manual selection works together

4. **Report Back**
   - Which mode worked better for your use case?
   - Any specific issues encountered?
   - FPS comparison?

---

## Quick Reference Commands

```bash
# Manual selection (default)
python src/cli_webcam.py --device mps --tracker-mode camera

# Auto-detection (if config.yaml has mode: "auto")
python src/cli_webcam.py --device mps

# Auto-detection (one-time test)
python src/cli_webcam.py --device mps --config configs/config_auto.yaml

# Kalman mode (automatic, no YOLO needed but no bbox selection)
python src/cli_webcam.py --device mps --tracker-mode kalman
```

---

## Next Steps After Testing

Once you've tested both modes, let me know:
1. **Which mode worked better for phones?**
2. **What was the FPS difference?**
3. **Any specific issues?** (detection failures, tracking drift, etc.)

I can then optimize the winner for your specific phone tracking use case!
