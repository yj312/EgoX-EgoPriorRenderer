import cv2
import mediapipe as mp
from mediapipe import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("Starting Pose test...")
base_options = python.BaseOptions(model_asset_path='data_preprocess/pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    num_poses=1,
    min_pose_detection_confidence=0.1,
    min_pose_presence_confidence=0.1,
    min_tracking_confidence=0.1
)

print("Reading image...")
img = cv2.imread('first_frame.jpg')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mp_image = Image(image_format=mp.ImageFormat.SRGB, data=rgb)

print("Processing image...")
with vision.PoseLandmarker.create_from_options(options) as landmarker:
    result = landmarker.detect(mp_image)
    if result.pose_landmarks:
        print('Pose detected successfully!')
        # NOSE is index 0
        nose = result.pose_landmarks[0][0]
        print('Nose:', nose.x, nose.y, nose.z)
    else:
        print('No pose detected.')

print("Done.")
