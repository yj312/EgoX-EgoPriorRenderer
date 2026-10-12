import json
import os
import sys
import cv2

def main():
    meta_path = "/home/i2slab4/yj/EgoX/datasets/kitchen1/meta.json"
    
    # 1. Read raw file content to handle invalid JSON placeholders
    print(f"Reading {meta_path}...")
    with open(meta_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Clean up non-JSON placeholders if present so we can parse it as JSON
    # First, let's try to parse it. If it fails, we will manually parse the structure or replace placeholders
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print("JSON contains placeholders. Cleaning up placeholders programmatically...")
        
        # Replace the placeholders with dummy values first to make it valid JSON
        cleaned = content
        
        # Replace camera_intrinsics placeholders
        cleaned = cleaned.replace("[fx,\n          0, cx\n        ]", "[1.0, 0.0, 1.0]")
        cleaned = cleaned.replace("[fx,\n          0, cx]", "[1.0, 0.0, 1.0]")
        cleaned = cleaned.replace("[\n          0, fy, cy\n        ]", "[0.0, 1.0, 1.0]")
        cleaned = cleaned.replace("[\n          0, fy, cy]", "[0.0, 1.0, 1.0]")
        
        # Direct string replacement for known placeholder patterns
        cleaned = cleaned.replace("fx", "1.0")
        cleaned = cleaned.replace("cx", "1.0")
        cleaned = cleaned.replace("fy", "1.0")
        cleaned = cleaned.replace("cy", "1.0")
        
        try:
            data = json.loads(cleaned)
            print("Successfully parsed cleaned JSON.")
        except json.JSONDecodeError as e2:
            print(f"Failed to parse cleaned JSON: {e2}")
            print("Fallback: Let's extract the ego_extrinsics block manually using string split.")
            
            # If everything fails, extract ego_extrinsics manually
            # Find the "ego_extrinsics": [ ... ] part
            ego_ext_start = content.find('"ego_extrinsics":')
            if ego_ext_start == -1:
                print("ERROR: Could not find ego_extrinsics in meta.json")
                sys.exit(1)
                
            # Extract everything from '"ego_extrinsics":' to the end
            ego_ext_part = content[ego_ext_start:]
            # Find the first '[' and matching closing ']'
            bracket_level = 0
            start_idx = ego_ext_part.find('[')
            end_idx = -1
            for idx in range(start_idx, len(ego_ext_part)):
                char = ego_ext_part[idx]
                if char == '[':
                    bracket_level += 1
                elif char == ']':
                    bracket_level -= 1
                    if bracket_level == 0:
                        end_idx = idx
                        break
            
            if end_idx == -1:
                print("ERROR: Could not balance brackets for ego_extrinsics")
                sys.exit(1)
                
            ego_ext_json = ego_ext_part[start_idx:end_idx+1]
            ego_extrinsics = json.loads(ego_ext_json)
            
            data = {
                "test_datasets": [
                    {
                        "exo_path": "./example/in_the_wild/videos/clean/exo1.mp4",
                        "ego_prior_path": "./example/in_the_wild/videos/clean/ego1_Prior.mp4",
                        "camera_intrinsics": [],
                        "camera_extrinsics": [],
                        "ego_intrinsics": [],
                        "ego_extrinsics": ego_extrinsics
                    }
                ]
            }
            print("Successfully extracted ego_extrinsics manually.")

    # 2. Get video width and height of exo1.mp4
    exo_rel_path = data["test_datasets"][0]["exo_path"]
    
    # Resolve the video absolute path relative to the EgoX project root (3 directories up from meta.json)
    meta_dir = os.path.dirname(os.path.abspath(meta_path))
    egox_root = os.path.dirname(os.path.dirname(meta_dir))
    
    if not os.path.isabs(exo_rel_path):
        exo_abs_path = os.path.abspath(os.path.join(egox_root, exo_rel_path))
    else:
        exo_abs_path = exo_rel_path
    
    print(f"Resolving absolute path of exo video: {exo_abs_path}")
    if not os.path.exists(exo_abs_path):
        print(f"ERROR: Video file does not exist at {exo_abs_path}")
        sys.exit(1)
        
    cap = cv2.VideoCapture(exo_abs_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    if w <= 0 or h <= 0:
        print("ERROR: Could not get video resolution, using fallback 1920x1080")
        w, h = 1920, 1080
        
    print(f"Detected video resolution: {w} x {h}")
    
    # 3. Calculate camera_intrinsics
    f = float(max(w, h))
    cx = float(w) / 2.0
    cy = float(h) / 2.0
    camera_intrinsics = [
        [f, 0.0, cx],
        [0.0, f, cy],
        [0.0, 0.0, 1.0]
    ]
    
    # 4. Set camera_extrinsics as Identity
    camera_extrinsics = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ]
    
    # 5. Set ego_intrinsics at 2880x2880 scale
    ego_intrinsics = [
        [1500.0, 0.0, 1440.0],
        [0.0, 1500.0, 1440.0],
        [0.0, 0.0, 1.0]
    ]
    
    # 6. Update data dictionary
    data["test_datasets"][0]["camera_intrinsics"] = camera_intrinsics
    data["test_datasets"][0]["camera_extrinsics"] = camera_extrinsics
    data["test_datasets"][0]["ego_intrinsics"] = ego_intrinsics
    
    # 7. Write back as beautifully formatted valid JSON
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        
    print(f"Successfully updated meta.json with:")
    print(f"  camera_intrinsics: {camera_intrinsics}")
    print(f"  camera_extrinsics: {camera_extrinsics}")
    print(f"  ego_intrinsics: {ego_intrinsics}")
    ego_ext_len = len(data['test_datasets'][0].get('ego_extrinsics', []))
    print(f"  ego_extrinsics: {ego_ext_len} frames")

if __name__ == "__main__":
    main()
