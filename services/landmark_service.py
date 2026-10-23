import os
import base64
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import Config

_detector = None
CANONICAL_ASPECT_RATIO = 16.0 / 9.0


def _resolve_hand_landmarker_model():
    candidates = [
        getattr(Config, "HAND_LANDMARKER_PATH", ""),
        os.path.join(os.path.dirname(mp.__file__), "models", "hand_landmarker.task"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Hand landmarker model not found. Checked: {candidates}"
    )


def get_hand_detector():
    global _detector
    if _detector is None:
        model_path = _resolve_hand_landmarker_model()
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            num_hands=2,
            min_hand_detection_confidence=0.40,
            min_hand_presence_confidence=0.40,
            min_tracking_confidence=0.40,
            running_mode=vision.RunningMode.IMAGE,
        )
        _detector = vision.HandLandmarker.create_from_options(options)
        print("MediaPipe HandLandmarker initialized.")
    return _detector


def extract_landmarks_from_frame(frame_bgr):
    """
    Extracts 126 landmarks (Left: 0..62, Right: 63..125) from a single BGR frame,
    normalizing non-16:9 camera aspect ratios to the 16:9 training frame convention,
    deduplicating overlapping boxes on the same physical wrist, and mapping true
    anatomical Left/Right hands accurately.
    Returns (landmarks_126, hand_detected_bool, hand_data_dict).
    """
    detector = get_hand_detector()
    h_px, w_px = frame_bgr.shape[:2]
    ar_scale = ((w_px / float(h_px)) / CANONICAL_ASPECT_RATIO) if h_px > 0 else 1.0

    img_rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)

    landmarks = np.zeros(126, dtype=np.float32)
    hand_dict = {"left_hand": None, "right_hand": None}

    if not result.hand_landmarks:
        return landmarks, False, hand_dict

    parsed = []
    for i, hand_lm in enumerate(result.hand_landmarks):
        label = "Right"
        score = 1.0
        if result.handedness and i < len(result.handedness) and len(result.handedness[i]) > 0:
            label = result.handedness[i][0].category_name
            score = float(result.handedness[i][0].score)

        pts = np.zeros((21, 3), dtype=np.float32)
        for j, lm in enumerate(hand_lm):
            x = float(lm.x) if np.isfinite(lm.x) else 0.0
            y = float(lm.y) if np.isfinite(lm.y) else 0.0
            z = float(lm.z) if np.isfinite(lm.z) else 0.0
            if abs(ar_scale - 1.0) > 1e-3:
                x = 0.5 + (x - 0.5) * ar_scale
                z = z * ar_scale
            pts[j] = (x, y, z)
        parsed.append((label, score, pts))

    # Suppress duplicate MediaPipe box on the exact same physical hand
    if len(parsed) == 2:
        w_dist = float(np.linalg.norm(parsed[0][2][0, :2] - parsed[1][2][0, :2]))
        if w_dist < 0.08:
            parsed = [parsed[0] if parsed[0][1] >= parsed[1][1] else parsed[1]]

    if len(parsed) == 2:
        # Unmirrored camera view: smaller x is signer's Right hand [63:126]; larger x is signer's Left hand [0:63]
        sorted_hands = sorted(parsed, key=lambda item: item[2][0, 0])
        assignments = [("right_hand", sorted_hands[0][2]), ("left_hand", sorted_hands[1][2])]
    else:
        label, _, pts = parsed[0]
        # MediaPipe Tasks HandLandmarker already reports anatomical Left/Right (unmirrored)
        slot_key = "left_hand" if label.lower() == "left" else "right_hand"
        assignments = [(slot_key, pts)]

    for slot_key, pts in assignments:
        base_idx = 63 if slot_key == "right_hand" else 0
        landmarks[base_idx:base_idx + 63] = pts.flatten()
        hand_dict[slot_key] = [
            {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in pts
        ]

    return landmarks, True, hand_dict


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
            z = float(lm.z) if hasattr(lm, "z") else float(lm.get("z", 0.0) or 0.0)
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
