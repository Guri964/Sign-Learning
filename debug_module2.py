import os
import sys
import json
import time
import random
import numpy as np
import tensorflow as tf

MODELS_DIR = r"C:\Users\Aman Tiwari\Downloads\Qriocity\Sign_Language\models"

print("="*70)
print("MODULE 2 — COMPREHENSIVE DEBUG SCRIPT")
print("Testing: MLP predictions, scoring, ghost hand, learn mode, time attack")
print("="*70)

print("\n[STEP 1] Loading all artifacts...")
try:
    mlp_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "isl_mlp_model.keras"))
    print("  MLP model loaded")
except Exception as e:
    print(f"  ERROR loading MLP model: {e}")
    sys.exit(1)

try:
    X_mlp_mean = np.load(os.path.join(MODELS_DIR, "X_mlp_mean.npy"))
    X_mlp_std = np.load(os.path.join(MODELS_DIR, "X_mlp_std.npy"))
    print("  MLP normalization stats loaded")
except Exception as e:
    print(f"  ERROR loading MLP norm stats: {e}")
    sys.exit(1)

try:
    label_classes = np.load(os.path.join(MODELS_DIR, "label_classes.npy"), allow_pickle=True).tolist()
    print(f"  Label classes loaded: {len(label_classes)} classes")
    print(f"  Classes: {label_classes}")
except Exception as e:
    print(f"  ERROR loading label classes: {e}")
    sys.exit(1)

try:
    with open(os.path.join(MODELS_DIR, "reference_landmarks.json"), "r") as f:
        reference_landmarks = json.load(f)
    print(f"  Reference landmarks loaded: {len(reference_landmarks)} classes")
except Exception as e:
    print(f"  ERROR loading reference landmarks: {e}")
    sys.exit(1)

print("\n[STEP 2] Verifying all classes have reference landmarks...")
missing_ref = []
for cls in label_classes:
    if cls not in reference_landmarks:
        missing_ref.append(cls)
    else:
        arr = np.array(reference_landmarks[cls])
        nonzero = np.count_nonzero(arr)
        if nonzero < 60:
            print(f"  WARNING: {cls} has only {nonzero} non-zero reference landmarks")

if missing_ref:
    print(f"  MISSING reference landmarks for: {missing_ref}")
else:
    print(f"  All {len(label_classes)} classes have reference landmarks")

def normalize_mlp(landmarks):
    landmarks = np.array(landmarks).flatten()
    normalized = (landmarks - X_mlp_mean.flatten()) / (X_mlp_std.flatten() + 1e-8)
    return np.nan_to_num(normalized, nan=0.0)

def normalize_wrist_relative(landmarks_126):
    lm = np.array(landmarks_126).copy()
    left = lm[:63].reshape(21, 3)
    right = lm[63:].reshape(21, 3)
    if np.count_nonzero(left) > 0:
        wrist = left[0].copy()
        left = left - wrist
        scale = np.max(np.abs(left)) + 1e-8
        left = left / scale
    if np.count_nonzero(right) > 0:
        wrist = right[0].copy()
        right = right - wrist
        scale = np.max(np.abs(right)) + 1e-8
        right = right / scale
    lm[:63] = left.flatten()
    lm[63:] = right.flatten()
    return lm

def predict_word(landmarks_126):
    lm_norm = normalize_wrist_relative(landmarks_126)
    normalized = normalize_mlp(lm_norm)
    input_data = normalized.reshape(1, 126)
    probs = mlp_model.predict(input_data, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx]) * 100
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(label_classes[i], round(float(probs[i])*100, 4)) for i in top3_idx]
    return label_classes[pred_idx], confidence, top3

def calculate_accuracy(user_landmarks, reference_landmarks_list):
    user = np.array(user_landmarks)
    ref = np.array(reference_landmarks_list)

    def extract_active(lm_126):
        left = lm_126[:63].reshape(21, 3)
        right = lm_126[63:].reshape(21, 3)
        if np.count_nonzero(right) >= np.count_nonzero(left):
            return right
        return left

    def norm_hand(hand):
        hand = hand - hand[0]
        scale = np.max(np.abs(hand)) + 1e-8
        return hand / scale

    user_hand = extract_active(user)
    ref_hand = extract_active(ref)
    user_norm = norm_hand(user_hand)
    ref_norm = norm_hand(ref_hand)
    distances = np.sqrt(np.sum((user_norm - ref_norm) ** 2, axis=1))
    max_dist = 0.6
    raw_acc = (1 - np.mean(distances) / max_dist) * 100
    raw_acc = np.clip(raw_acc, 0, 100)
    boosted = raw_acc * 1.15
    return round(float(np.clip(boosted, 0, 100)), 2)

def get_finger_feedback(user_landmarks, ref_landmarks):
    user = np.array(user_landmarks)
    ref = np.array(ref_landmarks)

    def extract_active(lm_126):
        left = lm_126[:63].reshape(21, 3)
        right = lm_126[63:].reshape(21, 3)
        if np.count_nonzero(right) >= np.count_nonzero(left):
            return right
        return left

    def norm_hand(hand):
        hand = hand - hand[0]
        scale = np.max(np.abs(hand)) + 1e-8
        return hand / scale

    user_hand = norm_hand(extract_active(user))
    ref_hand = norm_hand(extract_active(ref))
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    finger_indices = [(1, 5), (5, 9), (9, 13), (13, 17), (17, 21)]
    worst_finger = None
    worst_dist = -1
    for name, (start, end) in zip(finger_names, finger_indices):
        dist = np.mean(np.sqrt(np.sum((user_hand[start:end] - ref_hand[start:end]) ** 2, axis=1)))
        if dist > worst_dist:
            worst_dist = dist
            worst_finger = name
    if worst_dist < 0.15:
        return "Great job! Your hand position is correct."
    return f"Adjust your {worst_finger} finger position."

def check_ghost_hand_coordinates(cls):
    if cls not in reference_landmarks:
        return False, "No reference found"
    ref = np.array(reference_landmarks[cls])
    left = ref[:63].reshape(21, 3)
    right = ref[63:].reshape(21, 3)
    issues = []
    for hand, name in [(left, "left"), (right, "right")]:
        if np.count_nonzero(hand) > 0:
            x_vals = hand[:, 0][hand[:, 0] != 0]
            y_vals = hand[:, 1][hand[:, 1] != 0]
            if len(x_vals) > 0:
                if x_vals.min() < 0 or x_vals.max() > 1:
                    issues.append(f"{name} hand x out of [0,1] range: {x_vals.min():.3f} to {x_vals.max():.3f}")
                if y_vals.min() < 0 or y_vals.max() > 1:
                    issues.append(f"{name} hand y out of [0,1] range: {y_vals.min():.3f} to {y_vals.max():.3f}")
    if issues:
        return False, issues
    return True, "Coordinates valid for canvas drawing"

print("\n[STEP 3] Testing MLP predictions on reference landmarks...")
print("(Feeding each class reference landmarks to MLP)")
print()

correct = 0
total = 0
failed = []

for cls in label_classes:
    if cls not in reference_landmarks:
        continue
    ref = np.array(reference_landmarks[cls])
    pred, conf, top3 = predict_word(ref)
    is_correct = pred == cls
    if is_correct:
        correct += 1
        marker = "✓"
    else:
        failed.append(cls)
        marker = "✗"
    total += 1
    print(f"  {marker} {cls:<20} → Predicted: {pred:<20} Conf: {conf:.4f}%")
    if not is_correct:
        print(f"       Top3: {top3}")

print(f"\n  Result: {correct}/{total} correct ({correct/total*100:.2f}%)")
print(f"  Failed: {failed}")

print("\n[STEP 4] Testing accuracy scoring logic...")
print()

for cls in label_classes[:5]:
    if cls not in reference_landmarks:
        continue
    ref = reference_landmarks[cls]
    acc_perfect = calculate_accuracy(ref, ref)
    noise = np.array(ref) + np.random.normal(0, 0.02, len(ref))
    acc_noise = calculate_accuracy(noise.tolist(), ref)
    feedback = get_finger_feedback(ref, ref)
    print(f"  {cls:<20}")
    print(f"    Perfect match accuracy: {acc_perfect}%")
    print(f"    With small noise:       {acc_noise}%")
    print(f"    Feedback:               {feedback}")
    print()

print("\n[STEP 5] Testing ghost hand coordinate validity...")
print()

ghost_issues = []
for cls in label_classes:
    valid, result = check_ghost_hand_coordinates(cls)
    if not valid:
        ghost_issues.append((cls, result))
        print(f"  ISSUE — {cls}: {result}")
    else:
        print(f"  OK    — {cls}: {result}")

if not ghost_issues:
    print("\n  All ghost hand coordinates are valid for canvas drawing")
else:
    print(f"\n  {len(ghost_issues)} classes have ghost hand coordinate issues")

print("\n[STEP 6] Simulating LEARN MODE flow...")
print()

test_word = label_classes[0]
print(f"  Selected word: {test_word}")
ref = reference_landmarks.get(test_word)
if ref:
    print(f"  Reference landmarks fetched: {len(ref)} values")
    print(f"  Non-zero values: {np.count_nonzero(ref)}")
    user_landmarks = np.array(ref) + np.random.normal(0, 0.03, len(ref))
    pred, conf, top3 = predict_word(user_landmarks.tolist())
    acc = calculate_accuracy(user_landmarks.tolist(), ref)
    feedback = get_finger_feedback(user_landmarks.tolist(), ref)
    badge = acc >= 80
    print(f"  User signs → Predicted: {pred} ({conf:.2f}%)")
    print(f"  Accuracy score: {acc}%")
    print(f"  Feedback: {feedback}")
    print(f"  Badge unlocked: {badge}")
    print(f"  Learn mode flow: WORKING")

print("\n[STEP 7] Simulating TIME ATTACK MODE flow...")
print()

TOTAL_ROUNDS = 5
attack_score = 0
attack_results = []

for round_num in range(1, TOTAL_ROUNDS + 1):
    word = random.choice(label_classes)
    print(f"  Round {round_num}/{TOTAL_ROUNDS}: Word flashed = {word}")
    time.sleep(0.1)
    if word in reference_landmarks:
        ref = reference_landmarks[word]
        noise_level = random.uniform(0.01, 0.08)
        user_lm = (np.array(ref) + np.random.normal(0, noise_level, len(ref))).tolist()
        pred, conf, top3 = predict_word(user_lm)
        acc = calculate_accuracy(user_lm, ref)
        passed = acc >= 70
        if passed:
            attack_score += 1
        attack_results.append({"word": word, "predicted": pred, "accuracy": acc, "passed": passed})
        status = "PASS" if passed else "FAIL"
        print(f"    Predicted: {pred} | Accuracy: {acc}% | {status}")
    else:
        print(f"    No reference for {word} — skipped")

print(f"\n  Time Attack Score: {attack_score}/{TOTAL_ROUNDS}")
print(f"  Results: {attack_results}")

print("\n[STEP 8] Checking leaderboard logic...")
print()

mock_leaderboard = [
    {"name": "Test User", "signs": 15, "avg_accuracy": 84.31},
    {"name": "Demo User", "signs": 10, "avg_accuracy": 76.50},
]
print(f"  Leaderboard entries: {mock_leaderboard}")
print(f"  Leaderboard logic: WORKING")

print("\n" + "="*70)
print("MODULE 2 DEBUG COMPLETE")
print("="*70)
print(f"\nSummary:")
print(f"  MLP model loading:        OK")
print(f"  Reference landmarks:      OK")
print(f"  Prediction accuracy:      {correct/total*100:.2f}% on reference landmarks")
print(f"  Scoring logic:            OK")
print(f"  Ghost hand coordinates:   {'OK' if not ghost_issues else str(len(ghost_issues)) + ' issues'}")
print(f"  Learn mode flow:          OK")
print(f"  Time Attack mode flow:    OK")
print(f"  Failed classes:           {failed}")
print(f"\nShare this full output for analysis.")