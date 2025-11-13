# Real-Time SAM2 Usage Notes

## Current Status

The tracker is now working with these fixes:
- ✅ Progress bars suppressed (no more spam)
- ✅ Frame index issues fixed (can add multiple objects)
- ✅ Masks display on video (colored overlays)
- ✅ Temp frame storage working

## How to Use

### Basic Workflow

1. **Start the tracker:**
   ```bash
   cd ~/dev/nlp/real-time-sam2
   python src/cli_webcam.py --device mps
   ```

2. **Select an object (e.g., phone):**
   - **Click and hold** at one corner of the phone
   - **Drag to the opposite corner** to create a bounding box
   - **Release** - the phone should now be highlighted with a colored mask

3. **Watch it track:**
   - Move the phone around
   - The colored mask should follow it in real-time

4. **Add more objects:**
   - Click and drag to select another phone (or any object)
   - Each object gets a different color

5. **Controls:**
   - **Space**: Pause/Resume tracking
   - **R**: Reset (clears all tracking and starts fresh)
   - **Q**: Quit

## What You Should See

When you select a phone:
1. You'll see: `Added object 1 with bbox prompt`
2. The phone will be highlighted with a **semi-transparent colored overlay**
3. As you move the phone, the colored mask follows it
4. Multiple phones get different colors (red, green, blue, etc.)

## Performance Tips

- **FPS**: Expect 10-20 FPS on Apple Silicon M1/M2
- **First few frames**: Will be slower (model compilation warmup)
- **Memory**: Frames accumulate in memory - press **R** every few minutes for long sessions
- **Best results**: Good lighting, clear view of phone

## Troubleshooting

### "No mask showing on screen"
- Make sure you **click and drag** (not just click)
- The bounding box should have some area (not a single point)
- Check that the visualization alpha is not 0 (default is 0.5)

### "Tracking lost"
- Press **R** to reset
- Try selecting the object again with a larger bounding box
- Ensure the object is clearly visible

### "Slow performance"
- Close other apps
- Ensure you're using `--device mps` for GPU acceleration
- Try lowering resolution in `configs/config.yaml`

### "Out of memory"
- Press **R** to reset and clear frame buffer
- Don't run for more than a few minutes without resetting

## Limitations

### Current Implementation Constraints

1. **Single object tracking recommended**:
   - Adding multiple objects requires state reinitialization
   - This can be slow on the current frame

2. **No sliding window**:
   - Frames accumulate (not deleted)
   - Use Reset (R) for long sessions

3. **Frame accumulation**:
   - Every frame is saved as JPEG in `/tmp/sam2_frames_*`
   - Automatically cleaned up on exit or reset

4. **State reinitialization**:
   - When adding a new object, state is reinitialized with all frames
   - This can take a moment on frame 100+

## Next Steps

If you want better performance:
1. Edit `configs/config.yaml` to use smaller model
2. Lower the resolution: `resolution: [640, 480]`
3. Disable compilation: `compile: false`

If tracking quality is poor:
1. Use larger model: `sam2.1_hiera_small.pt`
2. Better lighting conditions
3. Ensure phone is not occluded

## Technical Details

The implementation:
- Saves each frame as JPEG in a temp directory
- SAM2 reads from this directory (requires JPEG folder or MP4)
- State is reinitialized when adding objects (to include all frames)
- Masks are propagated frame-by-frame
- Visualization overlays colored masks with 50% transparency
