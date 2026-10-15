import os
import json
import glob
from pathlib import Path

def compile_meta():
    # 1. Define paths (relative to EgoX-EgoPriorRenderer directory structure)
    renderer_root = Path(__file__).resolve().parents[2]
    # [ORIGINAL] 로컬 data_preprocess 경로 (주석 처리)
    # data_dir = renderer_root / "data_preprocess" / "data" / "0_48"
    # [NEW] /mnt/i2slab_data2/EgoExo4D_preprocessed 아래 data/0_48 사용
    data_dir = Path("/mnt/i2slab_data2/EgoExo4D_preprocessed/data/0_48")
    best_render_dir = data_dir / "best_ego_view_rendering"
    meta_files_dir = data_dir / "meta_files"
    
    # Target output location in EgoX
    dataset_name = os.environ.get("EGOX_DATASET_NAME", "egoexo4d_train1")
    # [ORIGINAL] egox_root-based path (commented out: home 파티션 용량 부족 우려)
    # egox_root = renderer_root.parent
    # output_meta_path = egox_root / "datasets" / dataset_name / "meta_compiled.json"
    # [NEW] /mnt 파티션에 저장 (EgoExo4D_preprocessed 하위)
    output_meta_path = Path("/mnt/i2slab_data2/EgoExo4D_preprocessed") / dataset_name / "meta_compiled.json"
    output_meta_final = Path("/mnt/i2slab_data2/EgoExo4D_preprocessed") / dataset_name / "meta.json"
    
    if not best_render_dir.exists():
        print(f"Error: Directory not found at {best_render_dir}")
        return

    compiled_entries = []
    
    # 2. Iterate through each completed take folder
    for take_path in sorted(best_render_dir.iterdir()):
        if not take_path.is_dir():
            continue
            
        take_name = take_path.name
        metadata_file = take_path / "metadata.json"
        if not metadata_file.exists():
            continue
            
        # 3. Read metadata.json to determine which camera was selected as 'best_camera'
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        best_camera = metadata.get("best_camera")
        if not best_camera:
            continue
            
        # 4. Find the matching full camera parameter file in meta_files
        meta_pattern = str(meta_files_dir / f"meta_{take_name}_{best_camera}_*.json")
        matching_files = glob.glob(meta_pattern)
        if not matching_files:
            print(f"Warning: No matching meta file found in meta_files/ for {take_name} (camera: {best_camera})")
            continue
            
        meta_file_path = matching_files[0]
        with open(meta_file_path, "r") as f:
            raw_meta = json.load(f)
            
        if "test_datasets" not in raw_meta or not raw_meta["test_datasets"]:
            continue
            
        # 5. Extract the single dataset entry containing the camera matrices
        entry = raw_meta["test_datasets"][0]
        
        # 6. Rewrite the paths to match the target datasets directory in EgoX
        # [NEW] /mnt 파티션 절대 경로
        exo_path = f"/mnt/i2slab_data2/EgoExo4D_preprocessed/{dataset_name}/videos/{take_name}/exo.mp4"
        ego_prior_path = f"/mnt/i2slab_data2/EgoExo4D_preprocessed/{dataset_name}/videos/{take_name}/ego_Prior.mp4"
        ego_gt_path = f"/mnt/i2slab_data2/EgoExo4D_preprocessed/{dataset_name}/videos/{take_name}/ego_GT.mp4"
        prompt = ""  # Prompt can be filled with caption.py later
        
        # Check if the video files exist
        if not Path(exo_path).exists() or not Path(ego_prior_path).exists() or not Path(ego_gt_path).exists():
            print(f"Skipping {take_name}: missing video files (exo: {Path(exo_path).exists()}, prior: {Path(ego_prior_path).exists()}, gt: {Path(ego_gt_path).exists()})")
            continue
            
        # Find matching vipe folder under vipe_results
        vipe_results_dir = data_dir / "vipe_results" / take_name
        vipe_pattern = str(vipe_results_dir / f"{best_camera}_video_{take_name}_{best_camera}_*")
        matching_vipe_folders = glob.glob(vipe_pattern)
        if not matching_vipe_folders:
            print(f"Warning: No matching vipe folder found for {take_name} (camera: {best_camera})")
            continue
        vipe_results_path = matching_vipe_folders[0]
        
        # Build meta dict for this take
        meta_dict = {
            "exo_video_path": exo_path,
            "ego_video_path": ego_gt_path,
            "ego_prior_path": ego_prior_path,
            "prompt": prompt,
            "take_name": take_name,
            "vipe_results_path": vipe_results_path,
            "best_camera": best_camera,
            "camera_extrinsics": entry["camera_extrinsics"],
            "camera_intrinsics": entry["camera_intrinsics"],
            "ego_extrinsics": entry["ego_extrinsics"],
            "ego_intrinsics": entry["ego_intrinsics"]
        }
        
        compiled_entries.append((take_name, meta_dict))
        print(f"Successfully processed {take_name} (Best Camera: {best_camera})")
        
    if not compiled_entries:
        print("No completed take data found to compile.")
        return
        
    # [NEW] dict-of-dicts format for wan_dataset.py
    output_meta = {take_name: meta_dict for take_name, meta_dict in compiled_entries}
    
    # 7. Write the consolidated meta_compiled.json and meta.json
    output_meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_meta_path, "w", encoding="utf-8") as f:
        json.dump(output_meta, f, indent=4, ensure_ascii=False)
    with open(output_meta_final, "w", encoding="utf-8") as f:
        json.dump(output_meta, f, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 Successfully compiled meta.json with {len(compiled_entries)} entries!")
    print(f"Saved to: {output_meta_path} and {output_meta_final}")

if __name__ == "__main__":
    compile_meta()
