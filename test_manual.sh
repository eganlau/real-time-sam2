#!/bin/bash
# Test Manual Phone Selection Mode

echo "========================================="
echo "Testing MANUAL phone selection mode"
echo "========================================="
echo ""
echo "Instructions:"
echo "  1. Position your phone in front of webcam"
echo "  2. Click and drag to select the phone"
echo "  3. Watch it track as you move the phone"
echo ""
echo "Controls:"
echo "  Mouse - Click and drag to select phone"
echo "  Space - Pause/Resume"
echo "  R     - Reset tracking"
echo "  Q     - Quit"
echo ""
echo "Starting in 3 seconds..."
sleep 3

python src/cli_webcam.py --device mps
