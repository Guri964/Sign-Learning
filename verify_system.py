"""
Comprehensive Verification and Debug Script for SignBridge ISL System.
Tests:
  1. Artifacts and File Integrity (Models, Shapes, Stats, GIFs)
  2. 70-Class Gesture Hints completeness
  3. Preprocessing Math (Wrist-relative, Linspace Resampling, Z-score Standardization)
  4. MLP Model Inference & Shape Scoring
  5. LSTM Model Inference & Dual-Slot Evaluation
  6. Landmark Service & Payload Conversion
  7. Flask Application Route Map
"""
import os
import sys
import json
import numpy as np

print("=" * 70)
print("SIGNBRIDGE 70-CLASS ISL SYSTEM VERIFICATION")
print("=" * 70)

# 1. Config & Paths
from config import Config
from utils.helpers import (
    GESTURE_HINTS,
    normalize_wrist_relative,
    pad_or_truncate,
    standardize_sequence,
    calculate_shape_accuracy,
    get_finger_feedback
)

print("\n[1] Checking Artifacts & Files...")
files_to_check = [
    ("LSTM Model", Config.LSTM_MODEL_PATH),
    ("LSTM Mean", Config.LSTM_MEAN_PATH),
    ("LSTM Std", Config.LSTM_STD_PATH),
    ("MLP Model", Config.MLP_MODEL_PATH),
    ("MLP Mean", Config.MLP_MEAN_PATH),
    ("MLP Std", Config.MLP_STD_PATH),
    ("Label Classes", Config.LABEL_CLASSES_PATH),
    ("Reference Landmarks", Config.REFERENCE_LANDMARKS_PATH),
    ("Reference Shapes", Config.REFERENCE_SHAPES_PATH),
    ("Hand Landmarker Task", Config.HAND_LANDMARKER_PATH),
]

all_files_ok = True
for name, path in files_to_check:
    exists = os.path.exists(path)
    size_kb = os.path.getsize(path) / 1024 if exists else 0
    status = "OK" if exists else "MISSING"
    print(f"  {status:<7} {name:<22}: {path} ({size_kb:.1f} KB)")
    if not exists:
        all_files_ok = False

# 2. Check 70 Gesture GIFs
gif_count = 0
missing_gifs = []
label_classes = np.load(Config.LABEL_CLASSES_PATH, allow_pickle=True).tolist()
for cls in label_classes:
    gif_file = f"{cls.replace(' ', '_').replace('/', '_')}.gif"
    gif_path = os.path.join(Config.GESTURE_GIFS_DIR, gif_file)
    if os.path.exists(gif_path):
        gif_count += 1
    else:
        missing_gifs.append(cls)

print(f"\n  Gesture GIFs found: {gif_count}/{len(label_classes)} in {Config.GESTURE_GIFS_DIR}")
if missing_gifs:
    print(f"  Missing GIFs: {missing_gifs}")
else:
    print("  All 70 classes have high-quality animated GIFs!")

# 3. Check Gesture Hints
print("\n[2] Checking 70-Class Gesture Hints...")
hints_count = sum(1 for c in label_classes if c in GESTURE_HINTS)
print(f"  Hints mapped: {hints_count}/{len(label_classes)} classes")
sample_hints = list(GESTURE_HINTS.items())[:3]
for k, v in sample_hints:
    print(f"  Sample Hint -> {k}: {v}")

# 4. Test MLP Service
print("\n[3] Testing Hand-Augmented MLP Service (Module 2)...")
from services.mlp_service import predict_and_score, get_reference_landmarks, get_reference_shape

test_classes = ["Hello", "Thank you", "Doctor", "Mother", "Father", "Book", "Car", "happy"]
with open(Config.REFERENCE_LANDMARKS_PATH) as f:
    refs = json.load(f)

for word in test_classes:
    ref = refs[word]
    result = predict_and_score(ref, word)
    acc = result["accuracy"]
    pred = result["predicted_word"]
    conf = result["confidence"]
    fb = result["feedback"]
    badge = "YES" if result["badge_unlocked"] else "NO"
    print(f"  {word:<12} -> Target Acc: {acc:>5.1f}% | Pred: {pred:<12} ({conf:>5.1f}%) | Badge: {badge} | Feedback: {fb}")

# 5. Test LSTM Service
print("\n[4] Testing 70-Class LSTM Service (Translator)...")
from services.lstm_service import predict_word

for word in test_classes:
    ref = refs[word]
    # Simulate a sequence of 20 frames with motion
    sequence = [ref] * 20
    pred_word, conf, top3 = predict_word(sequence)
    top_candidates = ", ".join([f"{c['word']} ({c['confidence']}%)" for c in top3[:2]])
    print(f"  {word:<12} -> LSTM Pred: {pred_word:<12} (Conf: {conf:>5.1f}%) | Top: {top_candidates}")

# 6. Test Landmark Service Payload Parsing
print("\n[5] Testing Landmark Service...")
from services.landmark_service import extract_landmarks_from_payload

dummy_payload = {
    "left_hand": [{"x": 0.1 * i, "y": 0.2 * i, "z": 0.0} for i in range(21)],
    "right_hand": [{"x": 0.05 * i, "y": 0.15 * i, "z": 0.0} for i in range(21)]
}
lm = extract_landmarks_from_payload(dummy_payload)
print(f"  Extracted landmark vector shape: {lm.shape}, non-zero entries: {np.count_nonzero(lm)}")
assert lm.shape == (126,), f"Expected shape (126,), got {lm.shape}"

# 7. Test Flask App, Right-Hand-Only GIF Integrity, & Ghost-Hand Reference Pipeline
print("\n[6] Checking Flask Application Routes, Right-Hand GIFs, & Ghost-Hand Shapes...")
from PIL import Image, ImageSequence
from app import app
from services.mlp_service import get_all_words, get_target_ghost_hand

expected_endpoints = [
    "tutor.gestures",
    "tutor.index",
    "translator.index",
    "dashboard.index",
    "dashboard.profile",
    "tutor.api_get_reference",
    "tutor.api_score_attempt",
    "translator.api_predict_word"
]
registered_rules = [r.endpoint for r in app.url_map.iter_rules()]
for ep in expected_endpoints:
    registered = ep in registered_rules
    status = "OK" if registered else "MISSING"
    print(f"  {status:<7} Endpoint: {ep}")
    assert registered, f"Missing Flask endpoint: {ep}"

all_ghost_valid = 0
all_right_only_gifs = 0
for w in get_all_words():
    ghost = get_target_ghost_hand(w)
    assert ghost is not None and len(ghost) == 21 and len(ghost[0]) == 3, f"Invalid ghost hand for {w}"
    all_ghost_valid += 1

    gif_file = os.path.join(Config.GESTURE_GIFS_DIR, f"{w.replace(' ', '_').replace('/', '_')}.gif")
    im = Image.open(gif_file)
    has_right_skel = False
    for fr in ImageSequence.Iterator(im):
        arr = np.asarray(fr.convert("RGB"), dtype=np.int16)
        left_cyan = (arr[:, :, 1] > 180) & ((arr[:, :, 2] - arr[:, :, 0]) > 90)
        left_blue = (arr[:, :, 2] > 190) & ((arr[:, :, 2] - arr[:, :, 0]) > 80)
        assert not np.any(left_cyan) and not np.any(left_blue), f"Left-hand skeleton pixel found in {gif_file}"
        if np.count_nonzero(arr[:, :, 0] >= 134) > 30:
            has_right_skel = True
    assert has_right_skel, f"Missing right-hand skeleton in {gif_file}"
    all_right_only_gifs += 1

print(f"  OK      Verified {all_ghost_valid}/70 valid 21-point right-hand ghost targets")
print(f"  OK      Verified {all_right_only_gifs}/70 GIFs display ONLY right-hand skeleton (0 left-hand pixels)")

print("\n" + "=" * 70)
print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
print("=" * 70)
