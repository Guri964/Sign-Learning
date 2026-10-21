import os
import io
import base64
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import Config

_detector = None

def get_hand_detector():
    global _detector
    if _detector is None:
        model_path = Config.HAND_LANDMARKER_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Hand landmarker model not found at {model_path}")
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.35,
            running_mode=vision.RunningMode.IMAGE
        )
        _detector = vision.HandLandmarker.create_from_options(options)
        print("MediaPipe HandLandmarker initialized.")
    return _detector

def extract_landmarks_from_frame(frame_bgr):
    """
    Extracts 126 landmarks (Left: 0..62, Right: 63..125) from a single BGR frame.
    Returns (landmarks_126, hand_detected_bool, hand_data_dict)
    """
    detector = get_hand_detector()
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)

    landmarks = np.zeros(126, dtype=np.float32)
    has_hand = False
    hand_dict = {"left_hand": None, "right_hand": None}

    if result.hand_landmarks and len(result.hand_landmarks) > 0:
        has_hand = True
        for i, hand_lm in enumerate(result.hand_landmarks):
            # Check handedness category
            # Note: In selfie/webcam mode, MediaPipe's "Left" is the user's left hand
            label = "Right"
            if result.handedness and i < len(result.handedness) and len(result.handedness[i]) > 0:
                label = result.handedness[i][0].category_name

            base_idx = 0 if label.lower() == "left" else 63
            lm_list = []
            for j, lm in enumerate(hand_lm):
                x = float(lm.x) if not np.isnan(lm.x) else 0.0
                y = float(lm.y) if not np.isnan(lm.y) else 0.0
                z = float(lm.z) if not np.isnan(lm.z) else 0.0
                landmarks[base_idx + j * 3] = x
                landmarks[base_idx + j * 3 + 1] = y
                landmarks[base_idx + j * 3 + 2] = z
                lm_list.append({"x": x, "y": y, "z": z})

            if label.lower() == "left":
                hand_dict["left_hand"] = lm_list
            else:
                hand_dict["right_hand"] = lm_list

    return landmarks, has_hand, hand_dict

def extract_landmarks_from_base64_frame(b64_string):
    """Decodes a base64 JPEG/PNG string and extracts 126 landmarks."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return np.zeros(126, dtype=np.float32), False, {"left_hand": None, "right_hand": None}
    return extract_landmarks_from_frame(frame)

def extract_sequence_from_frames(frames_b64_list):
    """
    Given a list of base64 frame strings, extracts a sequence of 126-d landmarks.
    """
    sequence = []
    for b64 in frames_b64_list:
        lm, detected, _ = extract_landmarks_from_base64_frame(b64)
        sequence.append(lm)
    return sequence

def extract_sequence_from_video_file(video_path):
    """
    Extracts 126-d landmarks for all frames in a video file (mp4/webm).
    """
    cap = cv2.VideoCapture(video_path)
    sequence = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        lm, _, _ = extract_landmarks_from_frame(frame)
        sequence.append(lm)
    cap.release()
    return sequence

def extract_landmarks_from_payload(hand_data):
    """
    Parses landmark dictionary from frontend {left_hand: [...], right_hand: [...]}.
    """
    landmarks = np.zeros(126, dtype=np.float32)
    if not hand_data:
        return landmarks
    left = hand_data.get("left_hand") or []
    right = hand_data.get("right_hand") or []
    if left and len(left) == 21:
        for idx, lm in enumerate(left):
            landmarks[idx * 3] = float(lm.get("x", 0.0) or 0.0)
            landmarks[idx * 3 + 1] = float(lm.get("y", 0.0) or 0.0)
            landmarks[idx * 3 + 2] = float(lm.get("z", 0.0) or 0.0)
    if right and len(right) == 21:
        for idx, lm in enumerate(right):
            landmarks[63 + idx * 3] = float(lm.get("x", 0.0) or 0.0)
            landmarks[63 + idx * 3 + 1] = float(lm.get("y", 0.0) or 0.0)
            landmarks[63 + idx * 3 + 2] = float(lm.get("z", 0.0) or 0.0)
    return np.nan_to_num(landmarks, nan=0.0)