import numpy as np
import tensorflow as tf
import json
import os

MODELS_DIR = r"C:\Users\Aman Tiwari\Downloads\Qriocity\Sign_Language\models"
SEQUENCE_LENGTH = 20

print("="*60)
print("SYNTHETIC LANDMARK DEBUG — NO WEBCAM NEEDED")
print("Testing model using stored reference landmarks")
print("="*60)

print("\nLoading model and artifacts...")
model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "isl_lstm_model.keras"))
X_mean = np.load(os.path.join(MODELS_DIR, "X_mean.npy"))
X_std = np.load(os.path.join(MODELS_DIR, "X_std.npy"))
label_classes = np.load(os.path.join(MODELS_DIR, "label_classes.npy"), allow_pickle=True).tolist()

with open(os.path.join(MODELS_DIR, "reference_landmarks.json"), "r") as f:
    reference_landmarks = json.load(f)

print(f"Model loaded. {len(label_classes)} classes.")
print(f"Reference landmarks available for: {list(reference_landmarks.keys())}")

def normalize_sequence(sequence):
    seq = np.array(sequence)
    normalized = (seq - X_mean) / (X_std + 1e-8)
    return np.nan_to_num(normalized, nan=0.0)

def predict_from_landmarks(landmarks_126):
    sequence = np.tile(landmarks_126, (SEQUENCE_LENGTH, 1))
    normalized = normalize_sequence(sequence)
    input_data = normalized.reshape(1, SEQUENCE_LENGTH, 126)
    probs = model.predict(input_data, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx]) * 100
    top5_idx = np.argsort(probs)[::-1][:5]
    top5 = [(label_classes[i], round(float(probs[i])*100, 1)) for i in top5_idx]
    return label_classes[pred_idx], confidence, top5

print("\n" + "="*60)
print("STEP 1 — PREDICTING FROM EXACT REFERENCE LANDMARKS")
print("(These are the averaged training landmarks per class)")
print("="*60)

correct = 0
total = 0
results = {}

for true_word in label_classes:
    if true_word not in reference_landmarks:
        print(f"  SKIP {true_word} — no reference landmarks found")
        continue

    ref = np.array(reference_landmarks[true_word])
    pred_word, confidence, top5 = predict_from_landmarks(ref)
    is_correct = pred_word == true_word

    if is_correct:
        correct += 1
        status = "CORRECT"
    else:
        status = "WRONG  "

    total += 1
    results[true_word] = {
        "predicted": pred_word,
        "confidence": confidence,
        "correct": is_correct,
        "top5": top5
    }

    marker = "✓" if is_correct else "✗"
    print(f"  {marker} [{status}] True: {true_word:<20} Predicted: {pred_word:<20} Conf: {confidence:.1f}%")
    if not is_correct:
        print(f"           Top5: {top5}")

print(f"\nExact reference accuracy: {correct}/{total} = {correct/total*100:.1f}%")

print("\n" + "="*60)
print("STEP 2 — PREDICTING WITH SMALL NOISE ADDED")
print("(Simulates slight hand position variation)")
print("="*60)

noise_correct = 0
noise_total = 0

for true_word in label_classes:
    if true_word not in reference_landmarks:
        continue

    ref = np.array(reference_landmarks[true_word])
    noise_results = []
    for trial in range(5):
        noise = np.random.normal(0, 0.02, ref.shape)
        noisy_ref = ref + noise
        pred_word, confidence, _ = predict_from_landmarks(noisy_ref)
        noise_results.append(pred_word == true_word)

    noise_acc = sum(noise_results) / len(noise_results) * 100
    if noise_acc >= 60:
        status = "STABLE"
    else:
        status = "UNSTABLE"

    noise_correct += sum(noise_results)
    noise_total += len(noise_results)
    print(f"  [{status}] {true_word:<20} Noise accuracy: {noise_acc:.0f}% ({sum(noise_results)}/5 trials)")

print(f"\nNoise robustness: {noise_correct}/{noise_total} = {noise_correct/noise_total*100:.1f}%")

print("\n" + "="*60)
print("STEP 3 — PREDICTING WITH SCALED LANDMARKS")
print("(Simulates different hand sizes / distances from camera)")
print("="*60)

scale_correct = 0
scale_total = 0

for true_word in label_classes:
    if true_word not in reference_landmarks:
        continue

    ref = np.array(reference_landmarks[true_word])
    scale_results = []
    for scale in [0.7, 0.85, 1.0, 1.15, 1.3]:
        scaled_ref = ref * scale
        pred_word, confidence, _ = predict_from_landmarks(scaled_ref)
        scale_results.append(pred_word == true_word)

    scale_acc = sum(scale_results) / len(scale_results) * 100
    if scale_acc >= 60:
        status = "STABLE"
    else:
        status = "UNSTABLE"

    scale_correct += sum(scale_results)
    scale_total += len(scale_results)
    print(f"  [{status}] {true_word:<20} Scale robustness: {scale_acc:.0f}% ({sum(scale_results)}/5 scales)")

print(f"\nScale robustness: {scale_correct}/{scale_total} = {scale_correct/scale_total*100:.1f}%")

print("\n" + "="*60)
print("STEP 4 — CROSS CONFUSION TEST")
print("(Feed word A landmarks, see if model mistakes it for word B)")
print("="*60)

print("\nMost confused pairs:")
confusion_pairs = []
for true_word in label_classes:
    if true_word not in reference_landmarks:
        continue
    ref = np.array(reference_landmarks[true_word])
    _, _, top5 = predict_from_landmarks(ref)
    if top5[0][0] != true_word:
        confusion_pairs.append((true_word, top5[0][0], top5[0][1]))
    elif len(top5) > 1 and top5[1][1] > 20:
        confusion_pairs.append((f"{true_word}(2nd)", top5[1][0], top5[1][1]))

for pair in confusion_pairs:
    print(f"  {pair[0]:<20} confused with {pair[1]:<20} ({pair[2]:.1f}%)")

print("\n" + "="*60)
print("FINAL DIAGNOSIS")
print("="*60)

exact_acc = correct/total*100 if total > 0 else 0
noise_pct = noise_correct/noise_total*100 if noise_total > 0 else 0
scale_pct = scale_correct/scale_total*100 if scale_total > 0 else 0

print(f"\nExact reference accuracy:  {exact_acc:.1f}%")
print(f"Noise robustness:          {noise_pct:.1f}%")
print(f"Scale robustness:          {scale_pct:.1f}%")

if exact_acc < 70:
    print("\nDIAGNOSIS: MODEL IS BIASED — needs retraining with class weights")
    print("ACTION: Retrain on Kaggle with balanced classes")
elif noise_pct < 60:
    print("\nDIAGNOSIS: MODEL IS FRAGILE — works on exact data but breaks with variation")
    print("ACTION: Retrain with more augmentation")
elif scale_pct < 60:
    print("\nDIAGNOSIS: MODEL IS SCALE SENSITIVE — hand distance from camera matters too much")
    print("ACTION: Normalize landmarks relative to wrist before feeding to model")
else:
    print("\nDIAGNOSIS: MODEL IS FINE — issue is with webcam landmark extraction or gesture quality")
    print("ACTION: Improve MediaPipe preprocessing pipeline")

print("\nShare this full output for next steps.")
