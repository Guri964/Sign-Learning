import os
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
            running_mode=vision.RunningMode.IMAGE,
        )
        _detector = vision.HandLandmarker.create_from_options(options)
        print("MediaPipe HandLandmarker initialized.")
    return _detector


def extract_landmarks_from_frame(frame_bgr):
    """
    Extracts 126 landmarks (Left: 0..62, Right: 63..125) from a single BGR frame.
    Returns (landmarks_126, hand_detected_bool, hand_data_dict).
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
            label = "Right"
            if result.handedness and i < len(result.handedness) and len(result.handedness[i]) > 0:
                label = result.handedness[i][0].category_name

            # Match unmirrored Holistic training convention: camera-left hand corresponds to signer's right hand
            slot_key = "right_hand" if label.lower() == "left" else "left_hand"
            if hand_dict[slot_key] is not None:
                slot_key = "left_hand" if slot_key == "right_hand" else "right_hand"
            base_idx = 63 if slot_key == "right_hand" else 0
            lm_list = []
            for j, lm in enumerate(hand_lm):
                x = float(lm.x) if np.isfinite(lm.x) else 0.0
                y = float(lm.y) if np.isfinite(lm.y) else 0.0
                z = float(lm.z) if np.isfinite(lm.z) else 0.0
                landmarks[base_idx + j * 3] = x
                landmarks[base_idx + j * 3 + 1] = y
                landmarks[base_idx + j * 3 + 2] = z
                lm_list.append({"x": x, "y": y, "z": z})

            hand_dict[slot_key] = lm_list

    return landmarks, has_hand, hand_dict


def extract_landmarks_from_base64_frame(b64_string):
    """Decodes a base64 JPEG/PNG string and extracts 126 landmarks."""
    if not isinstance(b64_string, str) or not b64_string.strip():
        return np.zeros(126, dtype=np.float32), False, {"left_hand": None, "right_hand": None}
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return np.zeros(126, dtype=np.float32), False, {"left_hand": None, "right_hand": None}
    return extract_landmarks_from_frame(frame)


def extract_sequence_from_frames(frames_b64_list):
    """Given a list of base64 frame strings, extracts a sequence of 126-d landmarks."""
    if not isinstance(frames_b64_list, list):
        raise ValueError("frames must be a list of base64-encoded image strings.")
    sequence = []
    for b64 in frames_b64_list:
        lm, _, _ = extract_landmarks_from_base64_frame(b64)
        sequence.append(lm)
    return sequence


def extract_sequence_from_video_file(video_path):
    """Extracts 126-d landmarks for all frames in a video file (mp4/webm)."""
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


def _parse_single_hand(hand_points) -> np.ndarray:
    out = np.zeros(63, dtype=np.float32)
    if not hand_points:
        return out
    if not isinstance(hand_points, (list, tuple, np.ndarray)) or len(hand_points) != 21:
        raise ValueError("Each detected hand must contain exactly 21 landmarks.")
    for idx, lm in enumerate(hand_points):
        if isinstance(lm, dict):
            x = float(lm.get("x", 0.0) or 0.0)
            y = float(lm.get("y", 0.0) or 0.0)
            z = float(lm.get("z", 0.0) or 0.0)
        elif isinstance(lm, (list, tuple, np.ndarray)) and len(lm) == 3:
            x, y, z = float(lm[0]), float(lm[1]), float(lm[2])
        else:
            raise ValueError("Landmark point must be an {x, y, z} object or [x, y, z] triple.")
        out[idx * 3] = x
        out[idx * 3 + 1] = y
        out[idx * 3 + 2] = z
    return out


def extract_landmarks_from_payload(hand_data) -> np.ndarray:
    """
    Parses a single frame landmark payload from the frontend or API client.
    Supports:
      - dict `{left_hand: [...21 pts...], right_hand: [...21 pts...]}`
      - flat numeric list/array of length 126
      - nested numeric array of shape (2, 21, 3) or (42, 3)
    Raises ValueError on malformed frame structures.
    """
    if hand_data is None:
        return np.zeros(126, dtype=np.float32)

    if isinstance(hand_data, dict):
        landmarks = np.zeros(126, dtype=np.float32)
        landmarks[:63] = _parse_single_hand(hand_data.get("left_hand"))
        landmarks[63:] = _parse_single_hand(hand_data.get("right_hand"))
        return np.nan_to_num(landmarks, nan=0.0, posinf=0.0, neginf=0.0)

    if isinstance(hand_data, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(hand_data, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("Numeric landmark frame contains non-numeric values.") from exc
        if arr.size == 0:
            return np.zeros(126, dtype=np.float32)
        if arr.size != 126:
            raise ValueError(f"Expected 126 landmark features per frame, got {arr.size}.")
        return np.nan_to_num(arr.reshape(126), nan=0.0, posinf=0.0, neginf=0.0)

    raise ValueError(f"Unsupported frame payload type: {type(hand_data).__name__}.")
