#!/bin/bash

# Set CUDA device
export CUDA_VISIBLE_DEVICES=2

# Video path
# VIDEO_PATH="/home/i2slab4/yj/EgoX/datasets/egobody4/videos/egobody4/exo.mp4"
# VIDEO_PATH="/home/i2slab4/yj/EgoX/EgoX-EgoPriorRenderer/example/in_the_wild/videos/cook2.mp4"
# VIDEO_PATH="../example/in_the_wild/videos/joker/exo.mp4"
# 숫자를 넣으면(예: 4) 해당 egobody 경로로 자동 지정, 경로를 직접 넣어도 동작
if [[ "$1" =~ ^[0-9]+$ ]]; then
    VIDEO_PATH="/home/i2slab4/yj/EgoX/datasets/egobody$1/videos/egobody$1/exo.mp4"
elif [[ -n "$1" ]]; then
    VIDEO_PATH="$1"
else
    VIDEO_PATH="/home/i2slab4/yj/EgoX/datasets/egobody4/videos/egobody4/exo.mp4"
fi


# Inference settings
START_FRAME=0
PIPELINE="lyra"

# Check if video file exists
if [[ ! -f "$VIDEO_PATH" ]]; then
    echo "ERROR: Video file not found: $VIDEO_PATH"
    exit 1
fi

# Detect frame count dynamically using Python OpenCV
TOTAL_FRAMES=$(python3 -c "import cv2; cap=cv2.VideoCapture('$VIDEO_PATH'); print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))" 2>/dev/null)
if [[ -z "$TOTAL_FRAMES" || "$TOTAL_FRAMES" -le 0 ]]; then
    echo "WARNING: Could not detect frame count, using default 49 frames."
    END_FRAME=48
else
    END_FRAME=$((TOTAL_FRAMES - 1))
fi
echo "=========================================="
echo "ViPE Inference"
echo "=========================================="
echo "Video: $VIDEO_PATH"
echo "Frame range: $START_FRAME to $END_FRAME (Total: $TOTAL_FRAMES frames)"
echo "Pipeline: $PIPELINE"
echo "=========================================="

# Run ViPE inference
vipe infer "$VIDEO_PATH" \
    --start_frame $START_FRAME \
    --end_frame $END_FRAME \
    --assume_fixed_camera_pose \
    --pipeline $PIPELINE

echo "Inference completed!"