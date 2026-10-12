#!/usr/bin/env python3
"""
Stitch PNG frames from best_ego_view_rendering folders (ego_GT and exo_GT)
into MP4 videos saved in the final EgoX dataset directory.
"""

import os
import json
from pathlib import Path

# Try to import imageio for robust video writing
try:
    import imageio
except ImportError:
    print("Error: 'imageio' package is not installed.")
    print("Please install it: pip install imageio[ffmpeg]")
    import sys
    sys.exit(1)

def stitch_videos():
    # 1. Define paths (relative to EgoX-EgoPriorRenderer directory structure)
    renderer_root = Path(__file__).resolve().parents[2]
    # [ORIGINAL] 로컬 data_preprocess 경로 (주석 처리)
    # data_dir = renderer_root / "data_preprocess" / "data" / "0_48"
    # [NEW] /mnt/i2slab_data2/EgoExo4D_preprocessed 아래 data/0_48 사용
    data_dir = Path("/mnt/i2slab_data2/EgoExo4D_preprocessed/data/0_48")
    best_render_dir = data_dir / "best_ego_view_rendering"
    
    # Target output location in EgoX
    dataset_name = os.environ.get("EGOX_DATASET_NAME", "egoexo4d_train1")
    # [ORIGINAL] egox_root-based path (commented out: home 파티션 용량 부족 우려)
    # egox_root = renderer_root.parent
    # target_videos_dir = egox_root / "datasets" / dataset_name / "videos"
    # [NEW] /mnt 파티션에 저장 (EgoExo4D_preprocessed 하위)
    target_videos_dir = Path("/mnt/i2slab_data2/EgoExo4D_preprocessed") / dataset_name / "videos"
    
    if not best_render_dir.exists():
        print(f"Error: Directory not found at {best_render_dir}")
        return

    print(f"Scanning for takes under: {best_render_dir}")
    print(f"Target outputs will be saved to: {target_videos_dir}\n")

    # 2. Iterate through each completed take folder
    for take_path in sorted(best_render_dir.iterdir()):
        if not take_path.is_dir():
            continue
            
        take_name = take_path.name
        metadata_file = take_path / "metadata.json"
        if not metadata_file.exists():
            print(f"Skipping {take_name}: No metadata.json found.")
            continue
            
        # 3. Read metadata.json to get the correct chronological frame list
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            
        frames_list = metadata.get("frames", [])
        if not frames_list:
            print(f"Skipping {take_name}: Empty frames array in metadata.json.")
            continue
            
        # 4. Define source and target paths
        ego_src_dir = take_path / "ego_GT"
        exo_src_dir = take_path / "exo_GT"
        
        target_take_dir = target_videos_dir / take_name
        target_take_dir.mkdir(parents=True, exist_ok=True)
        
        ego_dest_path = target_take_dir / "ego_GT.mp4"
        exo_dest_path = target_take_dir / "exo.mp4"
        
        # 5. Compile ego_GT frames into ego_GT.mp4
        if ego_src_dir.exists():
            print(f"Processing '{take_name}' -> ego_GT.mp4...")
            ego_frame_paths = []
            for frame_info in frames_list:
                filename = frame_info["filename"]
                frame_path = ego_src_dir / filename
                if frame_path.exists():
                    ego_frame_paths.append(frame_path)
                else:
                    print(f"  Warning: Missing frame image {frame_path}")
            
            if ego_frame_paths:
                try:
                    with imageio.get_writer(ego_dest_path, fps=30, quality=9, codec='libx264') as writer:
                        for path in ego_frame_paths:
                            img = imageio.imread(path)
                            writer.append_data(img)
                    print(f"  ✓ Saved: {ego_dest_path}")
                except Exception as e:
                    print(f"  ✗ Failed to save ego_GT.mp4: {e}")
        else:
            print(f"Warning: '{take_name}/ego_GT' directory not found.")
            
        # 6. Compile exo_GT frames into exo.mp4
        if exo_src_dir.exists():
            print(f"Processing '{take_name}' -> exo.mp4...")
            exo_frame_paths = []
            for frame_info in frames_list:
                filename = frame_info["filename"]
                frame_path = exo_src_dir / filename
                if frame_path.exists():
                    exo_frame_paths.append(frame_path)
                else:
                    print(f"  Warning: Missing frame image {frame_path}")
            
            if exo_frame_paths:
                try:
                    with imageio.get_writer(exo_dest_path, fps=30, quality=9, codec='libx264') as writer:
                        for path in exo_frame_paths:
                            img = imageio.imread(path)
                            writer.append_data(img)
                    print(f"  ✓ Saved: {exo_dest_path}")
                except Exception as e:
                    print(f"  ✗ Failed to save exo.mp4: {e}")
        else:
            print(f"Warning: '{take_name}/exo_GT' directory not found.")
            
        # 7. Copy ego_Prior.mp4 if it exists
        ego_prior_src = take_path / "ego_Prior" / "ego_Prior.mp4"
        ego_prior_dest = target_take_dir / "ego_Prior.mp4"
        if ego_prior_src.exists():
            print(f"Copying '{take_name}' -> ego_Prior.mp4...")
            try:
                import shutil
                shutil.copy2(ego_prior_src, ego_prior_dest)
                print(f"  ✓ Copied: {ego_prior_dest}")
            except Exception as e:
                print(f"  ✗ Failed to copy ego_Prior.mp4: {e}")
        else:
            print(f"Warning: '{take_name}/ego_Prior/ego_Prior.mp4' not found.")
            
        print("-" * 50)

    print("\n🎉 All frame stitching tasks are completed successfully!")

if __name__ == "__main__":
    stitch_videos()
