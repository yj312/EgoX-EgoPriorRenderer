#!/usr/bin/env bash

set -u

# Set CUDA device
export CUDA_VISIBLE_DEVICES=2

# Video path
VIDEO_PATH="/mnt/nas/i2slab_shared/HRI_EgoX/EgoX/datasets/egobody12/videos/egobody12/exo.mp4"
# VIDEO_PATH="../example/in_the_wild/videos/joker/exo.mp4"

# Inference settings
START_FRAME=0
PIPELINE="lyra"
FRAME_MODE="all"

usage() {
    echo "Usage: $0 [--frame 49|all]"
    echo "  --frame 49   Process frames 0~48 only."
    echo "  --frame all  Process all frames."
    echo "  No --frame   Same as --frame all."
}

# Parse frame option.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --frame|-f)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --frame requires a positive integer or 'all'." >&2
                usage >&2
                exit 2
            fi
            FRAME_MODE="$2"
            shift 2
            ;;
        --frame=*)
            FRAME_MODE="${1#*=}"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$FRAME_MODE" != "all" && ! "$FRAME_MODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --frame must be 'all' or a positive integer, e.g. 49." >&2
    exit 2
fi

# Check if video file exists
if [[ ! -f "$VIDEO_PATH" ]]; then
    echo "ERROR: Video file not found: $VIDEO_PATH"
    exit 1
fi

# Detect frame count dynamically using Python OpenCV.
TOTAL_FRAMES=$(python3 - "$VIDEO_PATH" <<'PY'
import sys
import cv2

cap = cv2.VideoCapture(sys.argv[1])
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
print(count)
PY
2>/dev/null)
if [[ -z "$TOTAL_FRAMES" || "$TOTAL_FRAMES" -le 0 ]]; then
    echo "ERROR: Could not detect frame count." >&2
    exit 1
fi

if [[ "$FRAME_MODE" == "all" ]]; then
    END_FRAME=$((TOTAL_FRAMES - 1))
else
    REQUESTED_FRAMES=$((10#$FRAME_MODE))
    if (( TOTAL_FRAMES < REQUESTED_FRAMES )); then
        echo "WARNING: Requested $REQUESTED_FRAMES frames, but video has $TOTAL_FRAMES. Processing all available frames."
        END_FRAME=$((TOTAL_FRAMES - 1))
    else
        END_FRAME=$((REQUESTED_FRAMES - 1))
    fi
fi
echo "=========================================="
echo "ViPE Inference"
echo "=========================================="
echo "Video: $VIDEO_PATH"
echo "Frame range: $START_FRAME to $END_FRAME (Total: $TOTAL_FRAMES frames)"
echo "Frame mode: $FRAME_MODE"
echo "Pipeline: $PIPELINE"
echo "=========================================="

# Run ViPE inference
if vipe infer "$VIDEO_PATH" \
    --start_frame "$START_FRAME" \
    --end_frame "$END_FRAME" \
    --assume_fixed_camera_pose \
    --pipeline "$PIPELINE"; then
    echo "Inference completed!"
else
    echo "Inference failed." >&2
    exit 1
fi

