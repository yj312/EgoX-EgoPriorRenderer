#!/usr/bin/env python3
"""
Automatically convert depth maps (.exr inside .zip) of the selected 'best_camera'
for each processed take into individual .npy files in the EgoX dataset directory.
"""

import os
import json
import zipfile
import glob
import re
import tempfile
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Try to import OpenEXR and Imath for .exr reading
try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False
    print("Warning: OpenEXR is not installed in the active environment.")
    print("Please install it: pip install OpenEXR")

def extract_frame_number(filename: str) -> int:
    """Extract frame number from filename."""
    stem = Path(filename).stem
    numbers = re.findall(r'\d+', stem)
    if numbers:
        return int(numbers[-1])
    return 0

def read_exr_depth(file_data: bytes) -> np.ndarray:
    """Read depth map from EXR file data using OpenEXR library."""
    if not HAS_OPENEXR:
        raise ImportError("OpenEXR library is required. Install with: pip install OpenEXR")
    
    with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as tmp_file:
        tmp_file.write(file_data)
        tmp_path = tmp_file.name
    
    try:
        exr_file = OpenEXR.InputFile(tmp_path)
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        channels = header['channels'].keys()
        depth_channel = None
        for ch_name in ['Z', 'depth', 'Depth', 'R', 'Y']:
            if ch_name in channels:
                depth_channel = ch_name
                break
        
        if depth_channel is None:
            depth_channel = list(channels)[0]
        
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        channel_str = exr_file.channel(depth_channel, FLOAT)
        
        depth_map = np.frombuffer(channel_str, dtype=np.float32)
        depth_map = depth_map.reshape((height, width))
        return depth_map
    finally:
        os.unlink(tmp_path)

def convert_depth_zip_to_npy(depth_zip_path: str, output_dir: str):
    """Convert EXR depth maps from zip archive to .npy files."""
    if not os.path.exists(depth_zip_path):
        raise FileNotFoundError(f"Depth zip file not found: {depth_zip_path}")
        
    os.makedirs(output_dir, exist_ok=True)
    
    with zipfile.ZipFile(depth_zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        depth_files = [f for f in file_list if f.endswith('.exr') and not os.path.basename(f).startswith('.')]
        
        if not depth_files:
            print(f"Warning: No EXR depth files found in {depth_zip_path}")
            return
            
        print(f"Found {len(depth_files)} EXR files in {depth_zip_path}. Converting...")
        for filename in tqdm(depth_files, desc="Converting depth maps"):
            try:
                with zip_ref.open(filename) as file:
                    file_data = file.read()
                
                depth_map = read_exr_depth(file_data)
                frame_num = extract_frame_number(filename)
                
                output_filename = f"depth_{frame_num:06d}.npy"
                output_path = os.path.join(output_dir, output_filename)
                np.save(output_path, depth_map)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

def main():
    # 1. Define paths (relative to EgoX-EgoPriorRenderer directory structure)
    renderer_root = Path(__file__).resolve().parents[2]
    # [ORIGINAL] 로컬 data_preprocess 경로 (주석 처리)
    # data_dir = renderer_root / "data_preprocess" / "data" / "0_48"
    # [NEW] /mnt/i2slab_data2/EgoExo4D_preprocessed 아래 data/0_48 사용
    data_dir = Path("/mnt/i2slab_data2/EgoExo4D_preprocessed/data/0_48")
    best_render_dir = data_dir / "best_ego_view_rendering"
    vipe_results_dir = data_dir / "vipe_results"
    
    # Target output location in EgoX
    dataset_name = os.environ.get("EGOX_DATASET_NAME", "egoexo4d_train1")
    # [ORIGINAL] egox_root-based path (commented out: home 파티션 용량 부족 우려)
    # egox_root = renderer_root.parent
    # output_depthmaps_base = egox_root / "datasets" / dataset_name / "depth_maps"
    # [NEW] /mnt 파티션에 저장 (EgoExo4D_preprocessed 하위)
    output_depthmaps_base = Path("/mnt/i2slab_data2/EgoExo4D_preprocessed") / dataset_name / "depth_maps"

    if not best_render_dir.exists():
        print(f"Error: Directory not found at {best_render_dir}")
        return

    if not HAS_OPENEXR:
        print("Error: OpenEXR is required to read EXR depth maps. Please run: pip install OpenEXR")
        return

    print("=" * 60)
    print("Auto-converting Best Camera Depth Maps from EXR (.zip) to NPY")
    print("=" * 60)
    print(f"vipe_results: {vipe_results_dir}")
    print(f"Output folder: {output_depthmaps_base}\n")

    # 2. Iterate through each completed take folder
    for take_path in sorted(best_render_dir.iterdir()):
        if not take_path.is_dir():
            continue
            
        take_name = take_path.name
        metadata_file = take_path / "metadata.json"
        if not metadata_file.exists():
            continue
            
        # 3. Read metadata.json to get best_camera
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        best_camera = metadata.get("best_camera")
        if not best_camera:
            continue
            
        # 4. Find the matching vipe_results folder for the best_camera
        take_vipe_dir = vipe_results_dir / take_name
        if not take_vipe_dir.exists():
            print(f"Warning: Vipe results folder for take {take_name} not found.")
            continue
            
        # Match subfolders starting with the best_camera name (e.g. cam02_*)
        matching_camera_dirs = list(take_vipe_dir.glob(f"{best_camera}_*"))
        if not matching_camera_dirs:
            print(f"Warning: No vipe directory found for {take_name} camera {best_camera}")
            continue
            
        camera_dir = matching_camera_dirs[0]
        depth_dir = camera_dir / "depth"
        
        # Look for the zip file (e.g. cam02.zip)
        zip_files = list(depth_dir.glob(f"{best_camera}.zip"))
        if not zip_files:
            # Fallback to any zip file in the depth folder if name doesn't match exactly
            zip_files = list(depth_dir.glob("*.zip"))
            
        if not zip_files:
            print(f"Warning: No zip file found in depth folder {depth_dir}")
            continue
            
        depth_zip = zip_files[0]
        
        # Output directory matches the take name under depth_maps
        output_dir = output_depthmaps_base / take_name
        
        print(f"Take: {take_name} | Best Camera: {best_camera}")
        print(f"  Zip Path: {depth_zip}")
        print(f"  Output Path: {output_dir}")
        
        try:
            convert_depth_zip_to_npy(str(depth_zip), str(output_dir))
        except Exception as e:
            print(f"  Error converting depth for {take_name}: {e}")
            
        print("-" * 60)

    print("\n🎉 All depth map conversions completed successfully!")

if __name__ == "__main__":
    main()
