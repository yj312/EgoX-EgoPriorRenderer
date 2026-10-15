#!/bin/bash
set -euo pipefail

# Set CUDA device
export CUDA_VISIBLE_DEVICES=2

usage() {
    echo "Usage: bash scripts/infer_vipe.sh [VIDEO_PATH|EGOBODY_ID] [--frame 49|all]"
    echo "  --frame 49   Process frames 0-48 (or all available frames if shorter)."
    echo "  --frame all  Process the entire video (default)."
    echo "  Input defaults to egobody4; options may appear before or after the input."
}

VIDEO_INPUT="4"
INPUT_SET=false
FRAME_MODE="all"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --frame)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --frame requires 49 or all." >&2
                exit 1
            fi
            FRAME_MODE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "ERROR: Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            if [[ "$INPUT_SET" == true ]]; then
                echo "ERROR: Specify only one video path or egobody ID." >&2
                exit 1
            fi
            VIDEO_INPUT="$1"
            INPUT_SET=true
            shift
            ;;
    esac
done

# A numeric input selects egobody<ID>; otherwise use the supplied video path.
if [[ "$VIDEO_INPUT" =~ ^[0-9]+$ ]]; then
    VIDEO_PATH="/home/i2slab4/yj/EgoX/datasets/egobody${VIDEO_INPUT}/videos/egobody${VIDEO_INPUT}/exo.mp4"
else
    VIDEO_PATH="$VIDEO_INPUT"
fi

# Inference settings
START_FRAME=0
PIPELINE="lyra"
FRAME_ARGS=(--start_frame "$START_FRAME")
case "$FRAME_MODE" in
    49)
        FRAME_ARGS+=(--end_frame 48)
        FRAME_RANGE="0 to 48 (up to 49 frames)"
        ;;
    all)
        # Omitting --end_frame lets ViPE read until the end of the video.
        FRAME_RANGE="0 to end of video (all frames)"
        ;;
    *)
        echo "ERROR: --frame must be 49 or all, got: $FRAME_MODE" >&2
        exit 1
        ;;
esac

# Check if video file exists
if [[ ! -f "$VIDEO_PATH" ]]; then
    echo "ERROR: Video file not found: $VIDEO_PATH"
    exit 1
fi

echo "=========================================="
echo "ViPE Inference"
echo "=========================================="
echo "Video: $VIDEO_PATH"
echo "Frame range: $FRAME_RANGE"
echo "Pipeline: $PIPELINE"
echo "=========================================="

# Run ViPE inference
vipe infer "$VIDEO_PATH" \
    "${FRAME_ARGS[@]}" \
    --assume_fixed_camera_pose \
    --pipeline $PIPELINE

echo "Inference completed!"
