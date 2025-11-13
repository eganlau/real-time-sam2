# Real-Time SAM2 Object Tracking

Real-time object segmentation and tracking using Meta's **Segment Anything Model 2 (SAM2)** with webcam and video file support. Optimized for Apple Silicon (MPS) with support for CUDA and CPU.

## Features

- **Real-time webcam tracking** with interactive object selection
- **Video file processing** with batch tracking capabilities
- **Multi-object tracking** with unique IDs and color coding
- **Automatic object detection** using YOLOv8 (optional)
- **Manual object selection** via click-and-drag interface
- **Performance optimization** with torch.compile and mixed precision
- **Flexible model selection** (tiny, small, base, large variants)
- **External webcam auto-detection** (prefers external over built-in)
- **Both library and CLI** for maximum flexibility

> **Note**: This implementation adapts SAM2's `VideoPredictor` for real-time streaming by maintaining a frame buffer. SAM2 was originally designed for batch video processing, so this streaming adapter accumulates frames and periodically manages the buffer for optimal performance.

## Project Structure

```
real-time-sam2/
├── src/
│   ├── realtime_sam2/          # Core library package
│   │   ├── __init__.py
│   │   ├── camera_tracker.py   # SAM2 camera predictor wrapper
│   │   ├── input_handler.py    # Webcam/video input management
│   │   ├── visualizer.py       # Mask overlay and rendering
│   │   ├── detector.py         # Optional YOLO integration
│   │   └── utils.py            # Utility functions
│   ├── cli_webcam.py           # Interactive webcam CLI
│   └── cli_video.py            # Video processing CLI
├── configs/
│   ├── config.yaml             # Main configuration
│   └── sam2.1_hiera_t_512.yaml # Fast 512x512 SAM2 config
├── requirements.txt
├── setup.py
└── README.md
```

## Requirements

- **Python**: 3.10+
- **PyTorch**: 2.5.1+ (with MPS/CUDA/CPU support)
- **Operating System**: macOS (Apple Silicon preferred), Linux, Windows
- **Optional**: CUDA-capable GPU for best performance, or Apple Silicon for MPS acceleration

## Installation

### Step 1: Install Miniconda (if not already installed)

**For macOS (Apple Silicon):**
```bash
# Download Miniconda for Apple Silicon
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# Install Miniconda
bash Miniconda3-latest-MacOSX-arm64.sh

# Follow the prompts and restart your terminal
```

**For Linux:**
```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install Miniconda
bash Miniconda3-latest-Linux-x86_64.sh
```

### Step 2: Create and Activate Conda Environment

```bash
# Create a new conda environment with Python 3.10
conda create --name sam2-realtime python=3.10 -y

# Activate the environment
conda activate sam2-realtime
```

### Step 3: Install PyTorch

**For Apple Silicon (MPS):**
```bash
conda install pytorch torchvision torchaudio -c pytorch
```

**For CUDA (NVIDIA GPU):**
```bash
# For CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

**For CPU only:**
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### Step 4: Clone and Install SAM2

```bash
# Navigate to your project directory
cd /path/to/your/projects

# Clone the Real-Time SAM2 repository
cd real-time-sam2

# Clone the official SAM2 repository
git clone https://github.com/facebookresearch/sam2.git
cd sam2

# Install SAM2
pip install -e .

# Go back to project root
cd ..
```

### Step 5: Download SAM2 Model Checkpoints

```bash
# Navigate to SAM2 checkpoints directory
cd sam2/checkpoints

# Download model checkpoints
./download_ckpts.sh

# This will download all SAM2.1 model variants:
# - sam2.1_hiera_tiny.pt (~38MB)
# - sam2.1_hiera_small.pt (~46MB)
# - sam2.1_hiera_base_plus.pt (~80MB)
# - sam2.1_hiera_large.pt (~224MB)

# Go back to project root
cd ../..
```

**Manual download** (if script fails):
```bash
cd sam2/checkpoints

# Tiny model (fastest, recommended for real-time)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

# Small model (good balance)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

# Base Plus model (higher quality)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

# Large model (best quality, slower)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

cd ../..
```

### Step 6: Install Real-Time SAM2 Package

```bash
# Install core dependencies
pip install -r requirements.txt

# Optional: Install with automatic detection support
pip install -r requirements.txt ultralytics

# Install the package in development mode
pip install -e .
```

### Step 7: Verify Installation

```bash
# Test PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Test MPS availability (macOS)
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Test CUDA availability (NVIDIA GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test SAM2 installation
python -c "from sam2.build_sam import build_sam2_video_predictor; print('SAM2 OK')"

# Test Real-Time SAM2 package
python -c "from realtime_sam2 import SAM2CameraTracker; print('Real-Time SAM2 OK')"
```

## Configuration

Edit `configs/config.yaml` to customize settings:

### Model Configuration
```yaml
model:
  # Choose model size: tiny (fastest), small, base_plus, large (best quality)
  checkpoint: "sam2/checkpoints/sam2.1_hiera_tiny.pt"

  # Config path (relative to SAM2 package, no 'sam2/' prefix)
  config: "configs/sam2.1/sam2.1_hiera_t.yaml"

  # Device: null (auto-detect), "mps", "cuda", "cpu"
  device: null

  # Enable torch.compile for speedup (recommended)
  compile: true

  # Use mixed precision (recommended for GPU/MPS)
  use_bfloat16: true
```

### Input Configuration
```yaml
input:
  # Prefer external webcam over built-in
  prefer_external_webcam: true

  # Webcam resolution
  resolution: [640, 480]  # Lower = faster
```

### Tracking Configuration
```yaml
tracking:
  # Mode: "manual", "auto", or "both"
  mode: "both"

  # Maximum objects to track
  max_objects: 10

  # Target classes for auto-detection
  target_classes:
    - "cell phone"
```

## Usage

### CLI: Interactive Webcam Tracking

```bash
# Basic usage (uses default config)
python src/cli_webcam.py

# Use custom config
python src/cli_webcam.py --config configs/config.yaml

# Force specific device
python src/cli_webcam.py --device mps
python src/cli_webcam.py --device cuda
python src/cli_webcam.py --device cpu

# Disable torch.compile (if having issues)
python src/cli_webcam.py --no-compile
```

**Keyboard Controls:**
- **Space**: Pause/Resume tracking
- **R**: Reset tracker
- **A**: Run auto-detection (if enabled)
- **Q**: Quit
- **Mouse**: Click and drag to select objects

### CLI: Video File Processing

```bash
# Process video with auto-detection
python src/cli_video.py input.mp4 output.mp4 --auto-detect

# Process with manual bounding boxes
python src/cli_video.py input.mp4 output.mp4 --bbox 100,100,200,200

# Track multiple objects
python src/cli_video.py input.mp4 output.mp4 \
  --bbox 100,100,200,200 \
  --bbox 300,150,400,250

# Use custom config
python src/cli_video.py input.mp4 output.mp4 \
  --auto-detect \
  --config configs/config.yaml \
  --device mps
```

### Python Library Usage

```python
from realtime_sam2 import SAM2CameraTracker, InputHandler, Visualizer
import cv2

# Initialize tracker
tracker = SAM2CameraTracker(
    config_file="configs/sam2.1/sam2.1_hiera_t.yaml",
    checkpoint_path="sam2/checkpoints/sam2.1_hiera_tiny.pt",
    device="mps",  # or "cuda", "cpu"
    use_compile=True,
    use_bfloat16=True
)

# Open webcam
input_handler = InputHandler(prefer_external=True)

# Initialize visualizer
visualizer = Visualizer(alpha=0.5, show_bbox=True, show_fps=True)

# Read first frame and initialize
ret, first_frame = input_handler.read()
tracker.initialize(first_frame)

# Add object to track (e.g., bounding box)
tracker.add_object(bbox=[100, 100, 200, 200])

# Main loop
while True:
    ret, frame = input_handler.read()
    if not ret:
        break

    # Track objects
    frame_idx, obj_ids, masks_dict = tracker.track(frame)

    # Visualize
    vis_frame = visualizer.visualize(frame, masks_dict, fps=30)

    # Display
    cv2.imshow("Tracking", vis_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
input_handler.release()
cv2.destroyAllWindows()
```

## Performance Optimization

### Model Selection

| Model | Size | Speed (Apple M1) | Quality | Use Case |
|-------|------|------------------|---------|----------|
| **Tiny** | 38MB | 15-25 FPS | Good | Real-time on CPU/MPS |
| **Small** | 46MB | 10-15 FPS | Better | Balanced performance |
| **Base+** | 80MB | 5-10 FPS | Great | High quality |
| **Large** | 224MB | 2-5 FPS | Best | Maximum quality |

### Optimization Tips

1. **Use smaller model**: Start with `tiny` or `small`
2. **Lower resolution**: Set `resolution: [640, 480]` in config
3. **Enable compilation**: Keep `compile: true` (after warmup)
4. **Use 512x512 config**: For 2-3x speedup
5. **Reduce frame skip**: Process every 2nd or 3rd frame
6. **Enable MPS**: For Apple Silicon acceleration

### Expected Performance

**Apple M1/M2 (MPS):**
- Tiny: 15-25 FPS @ 640x480
- Small: 10-15 FPS @ 640x480
- Base+: 5-10 FPS @ 640x480

**NVIDIA RTX 3090 (CUDA):**
- Tiny: 40-60 FPS @ 640x480
- Small: 30-40 FPS @ 640x480
- Base+: 20-30 FPS @ 640x480

**CPU (Intel i7/AMD Ryzen):**
- Tiny: 3-8 FPS @ 640x480
- Small: 2-5 FPS @ 640x480

## Troubleshooting

### Issue: "cannot import name 'build_sam2_camera_predictor'"
This is expected. The code uses `build_sam2_video_predictor` which is the correct SAM2 API. If you see this error, it means you're running old verification commands. Use the updated verification:
```bash
python -c "from sam2.build_sam import build_sam2_video_predictor; print('SAM2 OK')"
```

### Issue: "RuntimeError: You're likely running Python from the parent directory"
SAM2 doesn't allow running Python from the parent directory of the sam2 repository. Solutions:
```bash
# Option 1: Run from a different directory
cd ~ && python -c "from sam2.build_sam import build_sam2_video_predictor; print('SAM2 OK')"

# Option 2: Run from within the project directory (not its parent)
cd real-time-sam2/src && python -c "from sam2.build_sam import build_sam2_video_predictor; print('SAM2 OK')"
```

### Issue: "SAM2 is not installed"
```bash
cd sam2 && pip install -e . && cd ..
```

### Issue: "Model checkpoint not found"
```bash
cd sam2/checkpoints && ./download_ckpts.sh && cd ../..
```

### Issue: "MPS not available" (macOS)
- Ensure you have macOS 12.3+ and Apple Silicon (M1/M2/M3)
- Update PyTorch: `conda install pytorch torchvision -c pytorch`

### Issue: "torch.compile not working"
- Disable with `--no-compile` flag
- Or set `compile: false` in config

### Issue: "Slow performance"
1. Use smaller model (tiny)
2. Lower resolution (480p)
3. Use 512x512 config
4. Enable GPU/MPS acceleration
5. Check `frame_skip` setting

### Issue: "YOLO auto-detection not working"
```bash
pip install ultralytics
```

### Issue: "Webcam not detected"
```bash
# Test camera indices
python -c "import cv2; [print(f'Camera {i}: {cv2.VideoCapture(i).isOpened()}') for i in range(5)]"
```

## Advanced Usage

### Custom Object Detection

```python
from realtime_sam2.detector import ObjectDetector

# Initialize detector
detector = ObjectDetector(
    model_name="yolov8n.pt",
    confidence_threshold=0.5,
    target_classes=["cell phone", "person"],
    device="mps"
)

# Detect objects
detections = detector.detect(frame, return_format="bbox")

# Add detections to tracker
for det in detections:
    tracker.add_object(bbox=det['bbox'])
```

### Multi-Camera Support

```python
# Explicitly select camera
input_handler = InputHandler(source=1)  # Camera index 1

# Or use external camera preference
input_handler = InputHandler(prefer_external=True)
```

### Programmatic Configuration

```python
import yaml

# Load config
with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

# Modify settings
config['model']['checkpoint'] = 'sam2/checkpoints/sam2.1_hiera_small.pt'
config['tracking']['mode'] = 'auto'

# Use modified config
# ... (pass to tracker)
```

## Examples

### Example 1: Track Phone in Real-Time
```bash
# Auto-detect and track phone
python src/cli_webcam.py --config configs/config.yaml
# Press 'A' to run auto-detection
```

### Example 2: Process Video with Multiple Objects
```bash
python src/cli_video.py input.mp4 output.mp4 \
  --bbox 150,100,250,300 \
  --bbox 400,200,500,400
```

### Example 3: Fast Processing Mode
Edit `configs/config.yaml`:
```yaml
model:
  config: "configs/sam2.1/sam2.1_hiera_t.yaml"  # Use standard config
  checkpoint: "sam2/checkpoints/sam2.1_hiera_tiny.pt"
  compile: true
input:
  resolution: [640, 480]
performance:
  frame_skip: 2  # Process every 2nd frame
```

## Citation

If you use this project, please cite SAM2:

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```

## License

This project is licensed under the Apache License 2.0 - see the SAM2 repository for details.

## Acknowledgments

- **Meta AI Research** for Segment Anything Model 2
- **Ultralytics** for YOLOv8
- Reference implementations:
  - [zdata-inc/sam2_realtime](https://github.com/zdata-inc/sam2_realtime)
  - [Gy920/segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time)

## Support

For issues and questions:
- Open an issue on GitHub
- Check the [SAM2 repository](https://github.com/facebookresearch/sam2) for model-related questions
- Review the configuration file for available options

## Roadmap

- [ ] Web interface with Gradio
- [ ] REST API server
- [ ] Multi-threaded frame processing
- [ ] Mask export (JSON, PNG, COCO format)
- [ ] Video annotation tools
- [ ] Custom training support
- [ ] Docker deployment

---

**Made with SAM2 by Meta AI Research**
