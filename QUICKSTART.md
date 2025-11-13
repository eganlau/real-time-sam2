# Quick Start Guide

## For Testing on Your MacBook Pro

You're currently at the verification step. Here's how to proceed:

### Fix the Import Error

The error you encountered is because you need to run Python from a different directory (not the parent of `sam2`). Here's the corrected verification:

```bash
# Change to your home directory or anywhere outside real-time-sam2
cd ~

# Now test SAM2
python -c "from sam2.build_sam import build_sam2_video_predictor; print('SAM2 OK')"

# Test Real-Time SAM2 package
python -c "from realtime_sam2 import SAM2CameraTracker; print('Real-Time SAM2 OK')"
```

### Complete Setup Commands

If you haven't completed all installation steps yet, here's the full sequence:

```bash
# 1. Ensure you're in the project directory
cd ~/dev/nlp/real-time-sam2

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Optional: Install YOLO for auto-detection
pip install ultralytics

# 4. Install the package in development mode
pip install -e .

# 5. Verify installation from home directory
cd ~
python -c "from sam2.build_sam import build_sam2_video_predictor; print('SAM2 OK')"
python -c "from realtime_sam2 import SAM2CameraTracker; print('Real-Time SAM2 OK')"
```

### Run Your First Test

```bash
# Go back to project directory
cd ~/dev/nlp/real-time-sam2

# Run the webcam tracker
python src/cli_webcam.py --device mps

# Controls:
#   - Click and drag to select objects (e.g., your phone)
#   - Space: Pause/Resume
#   - R: Reset tracking
#   - A: Auto-detect (if you installed ultralytics)
#   - Q: Quit
```

### Common Issues & Fixes

#### Issue: "RuntimeError: You're likely running Python from the parent directory"
**Solution**: Don't run Python commands from `/Users/eganlau/dev/nlp/` - always run from `~/dev/nlp/real-time-sam2/` or from `~`

#### Issue: "cannot import name 'build_sam2_camera_predictor'"
**Solution**: This function doesn't exist. Use `build_sam2_video_predictor` instead (already fixed in the code)

#### Issue: "SAM2 is not installed"
**Solution**:
```bash
cd ~/dev/nlp/real-time-sam2/sam2
pip install -e .
cd ..
```

#### Issue: "Model checkpoint not found"
**Solution**:
```bash
cd ~/dev/nlp/real-time-sam2/sam2/checkpoints
./download_ckpts.sh
cd ../..
```

#### Issue: "Cannot find primary config" or "MissingConfigException"
**Solution**: The config path should be relative to SAM2 package (no `sam2/` prefix). Check that `configs/config.yaml` has:
```yaml
model:
  config: "configs/sam2.1/sam2.1_hiera_t.yaml"  # Correct
  # NOT: "sam2/configs/sam2.1/sam2.1_hiera_t.yaml"  # Wrong
```

### Quick Performance Test

```bash
# Test with smallest/fastest model (recommended for first run)
python src/cli_webcam.py --device mps

# If you want to try without compilation (simpler, but slower)
python src/cli_webcam.py --device mps --no-compile
```

### Expected Performance on M1/M2/M3

- **Tiny model**: 15-25 FPS @ 640x480 (default)
- **Small model**: 10-15 FPS @ 640x480
- First few frames will be slow (torch.compile warmup)
- After warmup, speed should stabilize

### Next Steps

Once you get it working:
1. Try tracking your phone by clicking and dragging a box around it
2. Test auto-detection by pressing 'A' (if you installed ultralytics)
3. Experiment with different model sizes in `configs/config.yaml`
4. Try processing a video file:
   ```bash
   python src/cli_video.py input.mp4 output.mp4 --auto-detect
   ```

## Need Help?

If you encounter issues:
1. Check the Troubleshooting section in README.md
2. Ensure you're running from the correct directory
3. Verify all dependencies are installed: `pip list | grep -E "torch|sam2|opencv"`
