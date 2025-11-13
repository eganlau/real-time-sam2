#!/bin/bash
# Test Automatic Phone Detection Mode

echo "========================================="
echo "Testing AUTOMATIC phone detection mode"
echo "========================================="
echo ""
echo "Prerequisites:"
echo "  pip install ultralytics"
echo ""
echo "Instructions:"
echo "  1. Just hold your phone in front of webcam"
echo "  2. YOLO will detect it automatically (1-2 sec)"
echo "  3. Watch it track without clicking!"
echo ""
echo "Controls:"
echo "  A     - Trigger detection manually (don't wait)"
echo "  Space - Pause/Resume"
echo "  R     - Reset tracking"
echo "  Q     - Quit"
echo ""
echo "Starting in 3 seconds..."
sleep 3

python src/cli_webcam.py --device mps --config configs/config_auto.yaml
