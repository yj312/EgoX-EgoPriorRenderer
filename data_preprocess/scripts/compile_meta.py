import os
import json
import glob
from pathlib import Path

def compile_meta():
    # 1. Define paths (relative to EgoX-EgoPriorRenderer directory structure)
    renderer_root = Path(__file__).resolve().parents[2]
    data_subdir = os.environ.get("DATA_SUBDIR", "877_925")
    data_dir = renderer_root / "data_preprocess" / "data" / data_subdir
    best_render_dir = data_dir / "best_ego_view_rendering"
    meta_files_dir = data_dir / "meta_files"
    
    # Target output location in EgoX
    dataset_name = os.environ.get("EGOX_DATASET_NAME", "egoexo4d_test1")
    egox_root = renderer_root.parent
    output_meta_path = egox_root / "datasets" / dataset_name / "meta_compiled.json"
    
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
        entry["exo_path"] = f"./datasets/{dataset_name}/videos/{take_name}/exo.mp4"
        entry["ego_prior_path"] = f"./datasets/{dataset_name}/videos/{take_name}/ego_Prior.mp4"
        entry["ego_gt_path"] = f"./datasets/{dataset_name}/videos/{take_name}/ego_GT.mp4"
        entry["prompt"] = ""  # Prompt can be filled with caption.py later
        
        compiled_entries.append(entry)
        print(f"Successfully processed {take_name} (Best Camera: {best_camera})")
        
    if not compiled_entries:
        print("No completed take data found to compile.")
        return
        
    output_meta = {
        "test_datasets": compiled_entries
    }
    
    # 7. Write the consolidated meta_compiled.json
    output_meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_meta_path, "w", encoding="utf-8") as f:
        json.dump(output_meta, f, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 Successfully compiled meta.json with {len(compiled_entries)} entries!")
    print(f"Saved to: {output_meta_path}")

if __name__ == "__main__":
    compile_meta()
