#!/usr/bin/env python3
"""
Automated Ego Extrinsics Estimation Script.
Estimates the user's head pose from Exo camera frames using MediaPipe FaceMesh
and OpenCV solvePnP, aligns translation scale using depth maps from vipe_results,
and populates the 'ego_extrinsics' field in meta.json.
"""

import os
import sys
import argparse
import json
import zipfile
import re
import tempfile
import logging
from pathlib import Path
import numpy as np
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Try to import MediaPipe
try:
    import mediapipe as mp
    from mediapipe import Image
    from mediapipe import ImageFormat
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

# Try to import OpenEXR and Imath for .exr reading
try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False

# Standard Canonical 3D Face Model coordinates (in millimeters)
# Origin [0, 0, 0] is at the nose tip.
# X points to the person's left (our right when facing them).
# Y points up.
CANONICAL_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (Pose ID: 0)
    (-150.0, 170.0, -100.0),     # Left eye inner (Pose ID: 1)
    (-225.0, 170.0, -135.0),     # Left eye outer (Pose ID: 3)
    (150.0, 170.0, -100.0),      # Right eye inner (Pose ID: 4)
    (225.0, 170.0, -135.0),      # Right eye outer (Pose ID: 6)
    (-150.0, -150.0, -125.0),    # Left mouth (Pose ID: 9)
    (150.0, -150.0, -125.0)      # Right mouth (Pose ID: 10)
], dtype=np.float32)

# MediaPipe Pose indices for standard model points
POSE_INDICES = [0, 1, 3, 4, 6, 9, 10]

# Default translation offset from head model origin (nose tip) to camera projection center
# (2 cm up, 1 cm forward in meters)
HEAD_TO_CAM_OFFSET = np.array([0.0, -0.02, 0.01], dtype=np.float32)

def matrix_to_quat(R):
    """Convert a 3x3 rotation matrix to a unit quaternion [qw, qx, qy, qz]."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz], dtype=np.float32)

def quat_to_matrix(q):
    """Convert a unit quaternion [qw, qx, qy, qz] to a 3x3 rotation matrix."""
    qw, qx, qy, qz = q
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if norm > 1e-6:
        qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    else:
        return np.eye(3, dtype=np.float32)
    
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy]
    ], dtype=np.float32)

def parse_matrix_string(matrix_str: str) -> np.ndarray:
    """Parse 12 floats from a string and reshape into a 3x4 W2C matrix."""
    floats = re.findall(r"[-+]?\d*\.\d+|\d+", matrix_str)
    if len(floats) != 12:
        raise ValueError(f"Expected 12 numbers in matrix, found {len(floats)}: {floats}")
    return np.array([float(x) for x in floats], dtype=np.float32).reshape(3, 4)

def interpolate_and_smooth_trajectory(translations, quaternions, is_valid, filter_window=5, sigma=3.0):
    """Interpolate gaps in trajectory and apply median and Gaussian filtering."""
    n = len(is_valid)
    valid_indices = [i for i, v in enumerate(is_valid) if v]
    
    if not valid_indices:
        logger.warning("No valid poses detected in the entire video. Using default poses.")
        default_c2w = np.eye(4, dtype=np.float32)
        default_c2w[2, 3] = 1.0  # 1m distance in Z
        default_w2c = np.linalg.inv(default_c2w)[:3, :]
        return [default_w2c.tolist() for _ in range(n)]
        
    # Boundary filling
    first_valid_idx = valid_indices[0]
    last_valid_idx = valid_indices[-1]
    
    # Create copies to prevent modifying raw data in place
    translations = translations.copy()
    quaternions = quaternions.copy()
    is_valid = is_valid.copy()
    
    for i in range(first_valid_idx):
        translations[i] = translations[first_valid_idx]
        quaternions[i] = quaternions[first_valid_idx]
        is_valid[i] = True
        
    for i in range(last_valid_idx + 1, n):
        translations[i] = translations[last_valid_idx]
        quaternions[i] = quaternions[last_valid_idx]
        is_valid[i] = True
        
    # Interpolation for gaps
    for idx in range(len(valid_indices) - 1):
        t1 = valid_indices[idx]
        t2 = valid_indices[idx+1]
        if t2 - t1 > 1:
            for t in range(t1 + 1, t2):
                alpha = (t - t1) / (t2 - t1)
                translations[t] = (1 - alpha) * translations[t1] + alpha * translations[t2]
                
                # NLERP for quaternions
                q1 = quaternions[t1]
                q2 = quaternions[t2]
                if np.dot(q1, q2) < 0:
                    q2 = -q2
                q_interp = (1 - alpha) * q1 + alpha * q2
                q_interp /= np.linalg.norm(q_interp)
                quaternions[t] = q_interp
                is_valid[t] = True

    # Median Filtering (to remove sudden tracking glitches/outliers)
    if filter_window > 1:
        translations = median_filter1d(translations, filter_window)
        quaternions = median_filter_quaternions(quaternions, filter_window)

    # Gaussian Smoothing (to remove high-frequency noise and tremor)
    if sigma > 0:
        translations, quaternions = gaussian_smooth_trajectory(translations, quaternions, sigma)
        
    # Convert back to W2C 3x4 matrices
    w2c_list = []
    for i in range(n):
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = quat_to_matrix(quaternions[i])
        c2w[:3, 3] = translations[i]
        w2c = np.linalg.inv(c2w)
        w2c_list.append(w2c[:3, :].tolist())
        
    return w2c_list

def median_filter1d(data, window_size=5):
    n, d = data.shape
    half = window_size // 2
    filtered = data.copy()
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        filtered[i] = np.median(data[start:end], axis=0)
    return filtered

def median_filter_quaternions(quaternions, window_size=5):
    n = len(quaternions)
    half = window_size // 2
    filtered = quaternions.copy()
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        
        q_ref = quaternions[i]
        q_window = quaternions[start:end].copy()
        for idx in range(len(q_window)):
            if np.dot(q_window[idx], q_ref) < 0:
                q_window[idx] = -q_window[idx]
                
        q_med = np.median(q_window, axis=0)
        norm = np.linalg.norm(q_med)
        if norm > 1e-6:
            filtered[i] = q_med / norm
        else:
            filtered[i] = q_ref
    return filtered

def gaussian_smooth_trajectory(translations, quaternions, sigma=3.0):
    n = len(translations)
    half_width = int(round(3 * sigma))
    x = np.arange(-half_width, half_width + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / np.sum(kernel)
    
    smoothed_translations = np.zeros_like(translations)
    smoothed_quaternions = np.zeros_like(quaternions)
    
    for i in range(n):
        start = max(0, i - half_width)
        end = min(n, i + half_width + 1)
        
        k_start = half_width - (i - start)
        k_end = half_width + (end - i)
        k = kernel[k_start:k_end]
        k = k / np.sum(k)
        
        smoothed_translations[i] = np.sum(translations[start:end] * k[:, None], axis=0)
        
        q_ref = quaternions[i]
        q_window = quaternions[start:end].copy()
        for idx in range(len(q_window)):
            if np.dot(q_window[idx], q_ref) < 0:
                q_window[idx] = -q_window[idx]
        
        q_avg = np.sum(q_window * k[:, None], axis=0)
        q_avg_norm = np.linalg.norm(q_avg)
        if q_avg_norm > 1e-6:
            q_avg /= q_avg_norm
        else:
            q_avg = q_ref
        smoothed_quaternions[i] = q_avg
        
    return smoothed_translations, smoothed_quaternions

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

def get_valid_depth(depth_map: np.ndarray, u: int, v: int, search_radius: int = 5) -> float:
    """Retrieve metric depth value with local search fallback if the coordinate is noisy/zero."""
    h, w = depth_map.shape
    d = depth_map[v, u]
    if np.isfinite(d) and 0.05 < d < 20.0:
        return float(d)
    
    # Neighborhood search
    best_d = None
    min_dist = float('inf')
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            ny, nx = v + dy, u + dx
            if 0 <= ny < h and 0 <= nx < w:
                val = depth_map[ny, nx]
                if np.isfinite(val) and 0.05 < val < 20.0:
                    dist = dx*dx + dy*dy
                    if dist < min_dist:
                        min_dist = dist
                        best_d = val
    if best_d is not None:
        return float(best_d)
    return float(d) if np.isfinite(d) else 1.0

def find_zip_entry(zip_file: zipfile.ZipFile, frame_idx: int) -> str | None:
    """Find the depth map file entry matching frame index inside zip file."""
    namelist = zip_file.namelist()
    for name in namelist:
        if name.endswith('.exr') and not os.path.basename(name).startswith('.'):
            basename = os.path.basename(name)
            numbers = re.findall(r'\d+', basename)
            if numbers and int(numbers[-1]) == frame_idx:
                return name
    return None

def estimate_extrinsics(
    dataset_name: str,
    video_subpath: str | None = None,
    filter_window: int = 5,
    sigma: float = 5.0,
    roll_adj: float = 0.0,
    pitch_adj: float = 0.0,
    yaw_adj: float = 0.0,
    max_rotation_jump: float = 45.0,
    max_translation_jump: float = 0.4,
    depth_tolerance: float = 0.5,
    ref_frame: int | None = None,
    ref_matrix: str | None = None
):
    """Run face analysis and update the meta.json file."""
    if not HAS_MEDIAPIPE:
        logger.error("MediaPipe is not installed. Please install: pip install mediapipe")
        return
    if not HAS_OPENEXR:
        logger.error("OpenEXR is not installed. Please install: pip install OpenEXR")
        return

    # 1. Define roots
    renderer_root = Path(__file__).resolve().parents[2]
    egox_root = Path(__file__).resolve().parents[3]
    
    dataset_dir = egox_root / "datasets" / dataset_name
    meta_json_path = dataset_dir / "meta.json"
    
    if not meta_json_path.exists():
        logger.error(f"meta.json not found at {meta_json_path}")
        return
        
    with open(meta_json_path, "r") as f:
        meta_data = json.load(f)
        
    if "test_datasets" not in meta_data or not meta_data["test_datasets"]:
        logger.error("No test_datasets found in meta.json")
        return
        
    entry = meta_data["test_datasets"][0]
    
    # 2. Resolve paths
    exo_rel_path = entry.get("exo_path")
    if not exo_rel_path:
        logger.error("No exo_path found in meta.json")
        return
        
    # Attempt to resolve the exo video path in a robust way
    possible_paths = [
        Path(exo_rel_path),
        renderer_root / exo_rel_path,
        egox_root / exo_rel_path,
        dataset_dir / "videos" / dataset_name / "exo.mp4",
        dataset_dir / "videos" / (video_subpath or "eating/exo.mp4"),
    ]
    
    exo_video_path = None
    for p in possible_paths:
        if p.exists():
            exo_video_path = p
            break
            
    if exo_video_path is None:
        # If none of the paths exist, log a warning and fallback to default representation
        exo_video_path = egox_root / exo_rel_path
        logger.warning(f"Could not find exo video at any of the expected paths: {[str(p) for p in possible_paths]}")
        
    logger.info(f"Using Exo Video: {exo_video_path}")
    
    # 3. Locate Depth zip file in vipe_results
    vipe_depth_dir = renderer_root / "vipe_results" / dataset_name / "depth"
    if not vipe_depth_dir.exists():
        logger.error(f"vipe_results depth directory not found at {vipe_depth_dir}")
        return
        
    zip_files = list(vipe_depth_dir.glob("*.zip"))
    if not zip_files:
        logger.error(f"No depth zip file found in {vipe_depth_dir}")
        return
        
    depth_zip_path = zip_files[0]
    logger.info(f"Using Depth Zip: {depth_zip_path}")
    
    # 4. Read Intrinsics from meta.json
    camera_intrinsics = np.array(entry["camera_intrinsics"], dtype=np.float32)
    logger.info(f"Loaded Exo Camera Intrinsics:\n{camera_intrinsics}")
    
    # Open video capture to read frames
    cap = cv2.VideoCapture(str(exo_video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video Info: {frame_width}x{frame_height}, {frame_count} frames (dynamically detected)")
    
    # Initialize MediaPipe PoseLandmarker
    model_path = renderer_root / "data_preprocess" / "pose_landmarker.task"
    if not model_path.exists():
        logger.error(f"Pose landmarker model not found at {model_path}")
        return
        
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        num_poses=1,
        min_pose_detection_confidence=0.1,
        min_pose_presence_confidence=0.1,
        min_tracking_confidence=0.1
    )
    
    raw_translations = np.zeros((frame_count, 3), dtype=np.float32)
    raw_quaternions = np.zeros((frame_count, 4), dtype=np.float32)
    is_valid = np.zeros(frame_count, dtype=bool)
    
    # Initialize quaternions to identity (looking forward)
    raw_quaternions[:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    # ========================================================================
    # PASS 1: Collect nose depth values for ALL frames to compute global
    # median depth.  This prevents the latch-up problem where a single
    # bad depth (background) is accepted as a new baseline after a gap>3,
    # poisoning subsequent outlier checks.
    # ========================================================================
    logger.info("=== Pass 1/2: Collecting nose depth values for global median depth computation ===")
    nose_depths_per_frame = {}  # frame_idx -> depth_value
    
    cap_pass1 = cv2.VideoCapture(str(exo_video_path))
    with vision.PoseLandmarker.create_from_options(options) as landmarker_pass1:
        with zipfile.ZipFile(depth_zip_path, 'r') as z_file_pass1:
            for t in range(frame_count):
                ret, frame_p1 = cap_pass1.read()
                if not ret:
                    continue
                rgb_frame_p1 = cv2.cvtColor(frame_p1, cv2.COLOR_BGR2RGB)
                mp_image_p1 = Image(image_format=ImageFormat.SRGB, data=rgb_frame_p1)
                det_p1 = landmarker_pass1.detect(mp_image_p1)
                
                if det_p1.pose_landmarks:
                    lm_nose_p1 = det_p1.pose_landmarks[0][0]
                    u_p1 = int(lm_nose_p1.x * frame_width)
                    v_p1 = int(lm_nose_p1.y * frame_height)
                    
                    depth_entry_p1 = find_zip_entry(z_file_pass1, t)
                    if depth_entry_p1:
                        try:
                            file_data_p1 = z_file_pass1.read(depth_entry_p1)
                            depth_map_p1 = read_exr_depth(file_data_p1)
                            dh_p1, dw_p1 = depth_map_p1.shape
                            u_d_p1 = max(0, min(dw_p1 - 1, int(round(u_p1 * (dw_p1 / frame_width)))))
                            v_d_p1 = max(0, min(dh_p1 - 1, int(round(v_p1 * (dh_p1 / frame_height)))))
                            d_p1 = get_valid_depth(depth_map_p1, u_d_p1, v_d_p1)
                            if np.isfinite(d_p1) and 0.05 < d_p1 < 20.0:
                                nose_depths_per_frame[t] = d_p1
                        except Exception:
                            pass
                
                if t % 50 == 0:
                    logger.info(f"  Pass 1 progress: frame {t}/{frame_count}")
    cap_pass1.release()
    
    # Compute global median depth
    if nose_depths_per_frame:
        all_depths = np.array(list(nose_depths_per_frame.values()))
        global_median_depth = float(np.median(all_depths))
        logger.info(f"Global median nose depth: {global_median_depth:.3f}m (from {len(all_depths)} frames, tolerance: ±{depth_tolerance}m)")
    else:
        global_median_depth = None
        logger.warning("Could not compute global median depth (no valid depths found in pass 1). Skipping depth constraint.")
    
    # ========================================================================
    # PASS 2: Main tracking loop with global median depth constraint
    # ========================================================================
    logger.info("=== Pass 2/2: Main tracking with global median depth constraint ===")
    
    # Re-initialize MediaPipe for pass 2
    base_options_p2 = python.BaseOptions(model_asset_path=str(model_path))
    options_p2 = vision.PoseLandmarkerOptions(
        base_options=base_options_p2,
        num_poses=1,
        min_pose_detection_confidence=0.1,
        min_pose_presence_confidence=0.1,
        min_tracking_confidence=0.1
    )
    
    # Re-open video capture for pass 2 (the original `cap` is at EOF)
    cap.release()
    cap = cv2.VideoCapture(str(exo_video_path))
    
    with vision.PoseLandmarker.create_from_options(options_p2) as landmarker:
        with zipfile.ZipFile(depth_zip_path, 'r') as z_file:
            for t in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Could not read frame {t}")
                    continue
                    
                # Process pose landmark detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
                detection_result = landmarker.detect(mp_image)
                
                estimated_successfully = False
                
                if detection_result.pose_world_landmarks and detection_result.pose_landmarks:
                    wl = detection_result.pose_world_landmarks[0]
                    landmarks = detection_result.pose_landmarks[0]
                    
                    # 1. Estimate R_ego directly from 3D world landmarks (eliminates solvePnP flips)
                    p_eye_l = np.array([wl[1].x, wl[1].y, wl[1].z], dtype=np.float32)
                    p_eye_r = np.array([wl[4].x, wl[4].y, wl[4].z], dtype=np.float32)
                    p_mouth_l = np.array([wl[9].x, wl[9].y, wl[9].z], dtype=np.float32)
                    p_mouth_r = np.array([wl[10].x, wl[10].y, wl[10].z], dtype=np.float32)
                    
                    eye_center = (p_eye_l + p_eye_r) / 2.0
                    mouth_center = (p_mouth_l + p_mouth_r) / 2.0
                    
                    head_up = eye_center - mouth_center
                    head_right = p_eye_r - p_eye_l
                    
                    head_forward = np.cross(head_right, head_up)
                    head_forward_norm = np.linalg.norm(head_forward)
                    
                    if head_forward_norm > 1e-6:
                        head_forward /= head_forward_norm
                        
                        head_right_ortho = np.cross(head_up, head_forward)
                        head_right_ortho_norm = np.linalg.norm(head_right_ortho)
                        
                        if head_right_ortho_norm > 1e-6:
                            head_right_ortho /= head_right_ortho_norm
                            head_up_ortho = np.cross(head_forward, head_right_ortho)
                            head_up_ortho /= np.linalg.norm(head_up_ortho)
                            
                            # Ego camera coordinates representation (X: right, Y: down, Z: forward)
                            R_ego = np.zeros((3, 3), dtype=np.float32)
                            R_ego[:, 0] = head_right_ortho
                            R_ego[:, 1] = -head_up_ortho
                            R_ego[:, 2] = -head_forward
                            
                            # Apply local rotation adjustments (pitch/yaw/roll)
                            if pitch_adj != 0.0 or roll_adj != 0.0 or yaw_adj != 0.0:
                                pitch_rad = np.radians(pitch_adj)
                                roll_rad = np.radians(roll_adj)
                                yaw_rad = np.radians(-yaw_adj)  # Positive yaw turns gaze left
                                
                                Rx = np.array([
                                    [1.0, 0.0, 0.0],
                                    [0.0, np.cos(pitch_rad), -np.sin(pitch_rad)],
                                    [0.0, np.sin(pitch_rad), np.cos(pitch_rad)]
                                ], dtype=np.float32)
                                
                                Ry = np.array([
                                    [np.cos(yaw_rad), 0.0, np.sin(yaw_rad)],
                                    [0.0, 1.0, 0.0],
                                    [-np.sin(yaw_rad), 0.0, np.cos(yaw_rad)]
                                ], dtype=np.float32)
                                
                                Rz = np.array([
                                    [np.cos(roll_rad), -np.sin(roll_rad), 0.0],
                                    [np.sin(roll_rad), np.cos(roll_rad), 0.0],
                                    [0.0, 0.0, 1.0]
                                ], dtype=np.float32)
                                
                                R_ego = R_ego @ Rz @ Ry @ Rx
                            
                            # 2. Extract Nose 2D coordinates for depth map query
                            lm_nose = landmarks[0]
                            u_nose = int(lm_nose.x * frame_width)
                            v_nose = int(lm_nose.y * frame_height)
                            
                            # Fetch corresponding depth map EXR file from zip
                            depth_entry_name = find_zip_entry(z_file, t)
                            if depth_entry_name:
                                try:
                                    file_data = z_file.read(depth_entry_name)
                                    depth_map = read_exr_depth(file_data)
                                    
                                    # Scale coordinates if depth map resolution differs
                                    dh, dw = depth_map.shape
                                    u_depth = int(round(u_nose * (dw / frame_width)))
                                    v_depth = int(round(v_nose * (dh / frame_height)))
                                    u_depth = max(0, min(dw - 1, u_depth))
                                    v_depth = max(0, min(dh - 1, v_depth))
                                    
                                    # Retrieve metric depth (in meters)
                                    d = get_valid_depth(depth_map, u_depth, v_depth)
                                    
                                    # === GLOBAL MEDIAN DEPTH CONSTRAINT ===
                                    # Reject depths that deviate too far from the global
                                    # median.  This prevents latch-up where a background
                                    # depth (e.g. 3m) is accepted as baseline after a gap,
                                    # then subsequent real depths (e.g. 1.5m) get rejected.
                                    if global_median_depth is not None:
                                        depth_dev = abs(d - global_median_depth)
                                        if depth_dev > depth_tolerance:
                                            logger.warning(
                                                f"Frame {t}: Depth {d:.3f}m deviates {depth_dev:.3f}m from "
                                                f"global median {global_median_depth:.3f}m (tolerance {depth_tolerance}m). "
                                                f"Replacing with median depth."
                                            )
                                            # Use the global median depth instead of throwing
                                            # away the entire frame – this preserves the
                                            # rotation estimate while correcting the position.
                                            d = global_median_depth
                                    
                                    # Back-project to 3D in Exo camera frame
                                    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
                                    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
                                    
                                    Z_ego = d
                                    X_ego = (u_nose - cx) * Z_ego / fx
                                    Y_ego = (v_nose - cy) * Z_ego / fy
                                    T_ego_nose = np.array([X_ego, Y_ego, Z_ego], dtype=np.float32)
                                    
                                    # Apply camera offset relative to the head frame
                                    T_ego = T_ego_nose + R_ego @ HEAD_TO_CAM_OFFSET
                                    
                                    # Outlier checking (reject large translation/rotation jumps compared to last valid frame)
                                    is_outlier = False
                                    if np.any(is_valid[:t]):
                                        prev_idx = np.where(is_valid[:t])[0][-1]
                                        frame_gap = t - prev_idx
                                        
                                        if frame_gap <= 3:
                                            # Scale thresholds with frame gap to accommodate natural fast movements over gaps
                                            max_dist_scaled = max_translation_jump * frame_gap
                                            max_angle_rad = np.radians(max_rotation_jump) * frame_gap
                                            
                                            # Translation check
                                            dist = np.linalg.norm(T_ego - raw_translations[prev_idx])
                                            if dist > max_dist_scaled:
                                                logger.warning(f"Frame {t}: Translation jump of {dist:.3f}m detected (limit {max_dist_scaled:.3f}m, gap {frame_gap}). Marking as invalid.")
                                                is_outlier = True
                                                
                                            # Rotation check
                                            q_curr = matrix_to_quat(R_ego)
                                            q_prev = raw_quaternions[prev_idx]
                                            cos_theta = abs(np.dot(q_curr, q_prev))
                                            cos_theta = min(1.0, max(0.0, cos_theta))
                                            angle_diff = 2.0 * np.arccos(cos_theta)
                                            if angle_diff > max_angle_rad:
                                                logger.warning(f"Frame {t}: Rotation jump of {np.degrees(angle_diff):.1f}° detected (limit {np.degrees(max_angle_rad):.1f}°, gap {frame_gap}). Marking as invalid.")
                                                is_outlier = True
                                        else:
                                            # Large tracking gap. Rather than keeping comparison to the distant anchor
                                            # (which would likely cause a rejection cascade), we reset the tracking baseline here.
                                            logger.info(f"Frame {t}: Large tracking gap of {frame_gap} frames. Resetting tracking baseline.")
                                    
                                    if not is_outlier:
                                        raw_translations[t] = T_ego
                                        raw_quaternions[t] = matrix_to_quat(R_ego)
                                        is_valid[t] = True
                                        estimated_successfully = True
                                except Exception as exr_err:
                                    logger.error(f"Error loading/processing depth map for frame {t}: {exr_err}")
                
                if not estimated_successfully:
                    logger.warning(f"Frame {t}: Pose/depth estimation failed or rejected.")
                        
    cap.release()
    
    # Process the entire trajectory: interpolate lost tracking and apply filtering/smoothing
    logger.info(f"Processing trajectory: interpolating missing frames, filter_window={filter_window}, sigma={sigma}...")
    ego_extrinsics_list = interpolate_and_smooth_trajectory(
        raw_translations, raw_quaternions, is_valid, filter_window=filter_window, sigma=sigma
    )
    
    # Apply reference frame alignment propagation if provided
    if ref_frame is not None and ref_matrix is not None:
        try:
            logger.info(f"Applying calibration propagation: aligning trajectory using reference frame {ref_frame}")
            w2c_user = parse_matrix_string(ref_matrix)
            
            if ref_frame >= len(ego_extrinsics_list):
                raise ValueError(f"Reference frame {ref_frame} is out of bounds for trajectory of size {len(ego_extrinsics_list)}")
                
            w2c_raw_ref = np.array(ego_extrinsics_list[ref_frame])
            
            def to_4x4(w2c):
                m = np.eye(4, dtype=np.float32)
                m[:3, :] = w2c
                return m
                
            w_user = to_4x4(w2c_user)
            w_raw_ref = to_4x4(w2c_raw_ref)
            
            # W_offset_inv = W_user * W_raw_ref^-1
            w_offset_inv = w_user @ np.linalg.inv(w_raw_ref)
            
            # Calibrate all frames in place
            calibrated_extrinsics = []
            for t in range(len(ego_extrinsics_list)):
                w_raw_t = to_4x4(np.array(ego_extrinsics_list[t]))
                w_cal_t = w_offset_inv @ w_raw_t
                calibrated_extrinsics.append(w_cal_t[:3, :].tolist())
                
            ego_extrinsics_list = calibrated_extrinsics
            logger.info("Successfully calibrated and propagated trajectory!")
        except Exception as e:
            logger.error(f"Failed to apply calibration propagation: {e}")
            raise e
            
    # 5. Populate and write back to meta.json
    # Save raw trajectory backup so that interactive tools can always
    # re-derive corrected trajectories without re-running estimation.
    entry["ego_extrinsics_raw"] = ego_extrinsics_list
    entry["ego_extrinsics"] = ego_extrinsics_list
    
    with open(meta_json_path, "w") as f:
        json.dump(meta_data, f, indent=4)
        
    logger.info(f"Successfully calculated and filled in {len(ego_extrinsics_list)} frames of ego_extrinsics!")
    logger.info(f"Raw trajectory backup saved as 'ego_extrinsics_raw' in meta.json.")
    logger.info(f"Updated meta.json saved to: {meta_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate and update ego_extrinsics in meta.json using head pose tracking.")
    parser.add_argument("--dataset_name", type=str, default="eating1", help="Target dataset name (e.g. eating1)")
    parser.add_argument("--video_subpath", type=str, default=None, help="Relative path of exo video under dataset folder")
    parser.add_argument("--filter_window", type=int, default=5, help="Median filter window size (default: 5)")
    parser.add_argument("--sigma", type=float, default=5.0, help="Gaussian smoothing sigma (default: 5.0)")
    parser.add_argument("--roll_adj", type=float, default=0.0, help="Roll camera angle adjustment in degrees (default: 0.0)")
    parser.add_argument("--pitch_adj", type=float, default=0.0, help="Pitch camera angle adjustment in degrees (default: 0.0)")
    parser.add_argument("--yaw_adj", type=float, default=0.0, help="Yaw camera angle adjustment in degrees (default: 0.0)")
    parser.add_argument("--max_rotation_jump", type=float, default=45.0, help="Max rotation jump threshold in degrees for consecutive frames (default: 45.0)")
    parser.add_argument("--max_translation_jump", type=float, default=0.4, help="Max translation jump threshold in meters for consecutive frames (default: 0.4)")
    parser.add_argument("--depth_tolerance", type=float, default=0.5, help="Max deviation from global median nose depth in meters (default: 0.5). Depths outside this range are clamped to the median to prevent latch-up.")
    parser.add_argument("--ref_frame", type=int, default=None, help="Reference frame index for calibration alignment propagation")
    parser.add_argument("--ref_matrix", type=str, default=None, help="Copy-pasted 3x4 W2C extrinsic matrix at the reference frame for propagation")
    args = parser.parse_args()
    
    estimate_extrinsics(
        args.dataset_name, 
        args.video_subpath, 
        args.filter_window, 
        args.sigma, 
        args.roll_adj, 
        args.pitch_adj, 
        args.yaw_adj,
        args.max_rotation_jump,
        args.max_translation_jump,
        args.depth_tolerance,
        args.ref_frame,
        args.ref_matrix
    )
