#!/bin/bash
# ============================================================================
# EgoX Preprocessing Pipeline Integration Script
# ============================================================================
# This script automates all 4 steps of EgoExo4D preprocessing:
# 1. infer_vipe_all_takes.sh (Render ego priors and GT frames)
# 2. compile_meta.py (Compile camera parameter metadata & generate meta.json)
# 3. convert_best_depths.py (Convert .exr depths to .npy files)
# 4. stitch_frames.py (Stitch frames into videos and copy ego_Prior.mp4)

# Exit immediately if any command fails
set -euo pipefail

# ----------------------------------------------------------------------------
# Configurations (Modify as needed)
# ----------------------------------------------------------------------------
# Name of the dataset folder created in EgoX/datasets/
export EGOX_DATASET_NAME="egoexo4d_test1"

# Conda installation path
CONDA_PROFILE_PATH="/home/i2slab4/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_NAME="egox-egoprior"

# ----------------------------------------------------------------------------
# Setup Conda Environment
# ----------------------------------------------------------------------------
echo "🌐 Setting up Conda environment..."
if [[ -f "$CONDA_PROFILE_PATH" ]]; then
    source "$CONDA_PROFILE_PATH"
    set +u
    conda activate "$CONDA_ENV_NAME"
    set -u
    echo "✅ Conda environment '$CONDA_ENV_NAME' activated successfully."
else
    echo "⚠️  WARNING: Conda profile script not found at $CONDA_PROFILE_PATH"
    echo "   Assuming '$CONDA_ENV_NAME' is already activated in your current shell."
fi

# Ensure we are in the correct repository directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "📂 Working directory: $SCRIPT_DIR"

# ----------------------------------------------------------------------------
# Step 1: Render ego priors and GT frames
# ----------------------------------------------------------------------------
echo ""
echo "============================================================================"
echo "🚀 STEP 1: Running infer_vipe_all_takes.sh..."
echo "============================================================================"
bash data_preprocess/scripts/infer_vipe_all_takes.sh

# ----------------------------------------------------------------------------
# Step 2: Compile metadata and generate meta.json
# ----------------------------------------------------------------------------
echo ""
echo "============================================================================"
echo "🚀 STEP 2: Running compile_meta.py..."
echo "============================================================================"
python3 data_preprocess/scripts/compile_meta.py

# Create target meta.json for EgoX model inference from the compiled meta
compiled_json="../datasets/${EGOX_DATASET_NAME}/meta_compiled.json"
target_json="../datasets/${EGOX_DATASET_NAME}/meta.json"

if [[ -f "$compiled_json" ]]; then
    cp "$compiled_json" "$target_json"
    echo "✅ Successfully copied '$compiled_json' -> '$target_json'"
else
    echo "❌ ERROR: Compiled meta file not found at: $compiled_json"
    exit 1
fi

# ----------------------------------------------------------------------------
# Step 3: Convert .exr depths to .npy files
# ----------------------------------------------------------------------------
echo ""
echo "============================================================================"
echo "🚀 STEP 3: Running convert_best_depths.py..."
echo "============================================================================"
python3 data_preprocess/scripts/convert_best_depths.py

# ----------------------------------------------------------------------------
# Step 4: Stitch frames into videos and copy ego_Prior
# ----------------------------------------------------------------------------
echo ""
echo "============================================================================"
echo "🚀 STEP 4: Running stitch_frames.py..."
echo "============================================================================"
python3 data_preprocess/scripts/stitch_frames.py

echo ""
echo "🎉 [Pipeline Complete] All preprocessing steps completed successfully!"
echo "📍 Final dataset output: ../datasets/${EGOX_DATASET_NAME}/"
