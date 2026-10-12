import cv2
import numpy as np
import mediapipe as mp
from mediapipe import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def test():
    cap = cv2.VideoCapture('/home/i2slab4/yj/EgoX/datasets/eating1/videos/eating/exo.mp4')
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read frame")
        return
        
    model_path = '/home/i2slab4/yj/EgoX/EgoX-EgoPriorRenderer/data_preprocess/pose_landmarker.task'
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        num_poses=1
    )
    
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        res = landmarker.detect(mp_image)
        
        if res.pose_world_landmarks:
            wl = res.pose_world_landmarks[0]
            print("Successfully retrieved pose_world_landmarks!")
            
            # Print nose, eyes, mouth corners
            for idx in [0, 1, 4, 9, 10]:
                print(f"Landmark {idx}: ({wl[idx].x:.4f}, {wl[idx].y:.4f}, {wl[idx].z:.4f})")
                
            # Compute vectors
            p_nose = np.array([wl[0].x, wl[0].y, wl[0].z])
            p_eye_l = np.array([wl[1].x, wl[1].y, wl[1].z])
            p_eye_r = np.array([wl[4].x, wl[4].y, wl[4].z])
            p_mouth_l = np.array([wl[9].x, wl[9].y, wl[9].z])
            p_mouth_r = np.array([wl[10].x, wl[10].y, wl[10].z])
            
            eye_center = (p_eye_l + p_eye_r) / 2.0
            mouth_center = (p_mouth_l + p_mouth_r) / 2.0
            
            head_up = eye_center - mouth_center
            head_right = p_eye_r - p_eye_l
            
            head_forward = np.cross(head_right, head_up)
            head_forward /= np.linalg.norm(head_forward)
            
            head_right_ortho = np.cross(head_up, head_forward)
            head_right_ortho /= np.linalg.norm(head_right_ortho)
            
            head_up_ortho = np.cross(head_forward, head_right_ortho)
            head_up_ortho /= np.linalg.norm(head_up_ortho)
            
            # R_ego (X: -right, Y: -up, Z: forward)
            R_ego = np.zeros((3, 3))
            R_ego[:, 0] = -head_right_ortho
            R_ego[:, 1] = -head_up_ortho
            R_ego[:, 2] = head_forward
            
            print("Calculated R_ego:\n", R_ego)
            print("Determinant:", np.linalg.det(R_ego))
        else:
            print("No pose_world_landmarks detected")

if __name__ == '__main__':
    test()
