import cv2
import numpy as np
import tensorflow as tf
import json
import os
import time
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
import urllib.request

MODELS_DIR = r"C:\Users\Aman Tiwari\Downloads\Qriocity\Sign_Language\models"
SEQUENCE_LENGTH = 20

print("="*60)
print("WEBCAM DEBUG — ISL LSTM MODEL")
print("="*60)

print("\nLoading model and artifacts...")
model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "isl_lstm_model.keras"))
X_mean = np.load(os.path.join(MODELS_DIR, "X_mean.npy"))
X_std = np.load(os.path.join(MODELS_DIR, "X_std.npy"))
label_classes = np.load(os.path.join(MODELS_DIR, "label_classes.npy"), allow_pickle=True).tolist()
print(f"Model loaded. Classes: {label_classes}")

model_path = os.path.join(MODELS_DIR, "hand_landmarker.task")
if not os.path.exists(model_path):
    print("Downloading MediaPipe hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )
    print("Downloaded.")

options = HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
    running_mode=vision.RunningMode.IMAGE
)
detector = HandLandmarker.create_from_options(options)
print("MediaPipe loaded.")

def extract_landmarks(frame_bgr):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)
    landmarks = np.zeros(126)
    if result.hand_landmarks:
        for i, hand_lm in enumerate(result.hand_landmarks[:2]):
            base = i * 63
            for j, lm in enumerate(hand_lm):
                landmarks[base + j*3] = lm.x if not np.isnan(lm.x) else 0.0
                landmarks[base + j*3+1] = lm.y if not np.isnan(lm.y) else 0.0
                landmarks[base + j*3+2] = lm.z if not np.isnan(lm.z) else 0.0
    return landmarks, result.hand_landmarks is not None and len(result.hand_landmarks) > 0

def normalize_sequence(sequence):
    seq = np.array(sequence)
    normalized = (seq - X_mean) / (X_std + 1e-8)
    return np.nan_to_num(normalized, nan=0.0)

def predict_from_sequence(sequence):
    if len(sequence) < 5:
        return None, 0, []
    if len(sequence) >= SEQUENCE_LENGTH:
        indices = np.linspace(0, len(sequence)-1, SEQUENCE_LENGTH, dtype=int)
        padded = np.array([sequence[i] for i in indices])
    else:
        padded = np.array(sequence)
        pad_width = SEQUENCE_LENGTH - len(padded)
        padded = np.vstack([padded, np.zeros((pad_width, 126))])
    normalized = normalize_sequence(padded)
    input_data = normalized.reshape(1, SEQUENCE_LENGTH, 126)
    probs = model.predict(input_data, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx]) * 100
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(label_classes[i], round(float(probs[i])*100, 1)) for i in top3_idx]
    return label_classes[pred_idx], confidence, top3

frame_buffer = deque(maxlen=SEQUENCE_LENGTH * 2)
last_prediction = "—"
last_confidence = 0
last_top3 = []
last_pred_time = time.time()
PREDICT_EVERY = 1.5
hand_detected = False
prediction_count = defaultdict = {}
for cls in label_classes:
    prediction_count[cls] = 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n" + "="*60)
print("WEBCAM STARTED")
print("Controls:")
print("  SPACE — force predict now")
print("  C     — clear buffer")
print("  S     — show prediction summary")
print("  Q     — quit")
print("="*60)
print("\nSign each word and watch what the model predicts.")
print("Model predicts automatically every 1.5 seconds.\n")

from collections import defaultdict
prediction_history = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display = frame.copy()

    landmarks, detected = extract_landmarks(frame)
    hand_detected = detected

    if detected:
        frame_buffer.append(landmarks)

    now = time.time()
    if now - last_pred_time >= PREDICT_EVERY and len(frame_buffer) >= 5:
        sequence = list(frame_buffer)
        pred, conf, top3 = predict_from_sequence(sequence)
        if pred:
            last_prediction = pred
            last_confidence = conf
            last_top3 = top3
            last_pred_time = now
            prediction_count[pred] = prediction_count.get(pred, 0) + 1
            prediction_history.append((pred, conf))
            print(f"PREDICTED: {pred:<20} Confidence: {conf:.1f}%  |  Top3: {top3}")

    h, w = display.shape[:2]

    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (5, 8, 16), -1)
    cv2.addWeighted(overlay, 0.8, display, 0.2, 0, display)

    hand_color = (0, 255, 100) if hand_detected else (100, 100, 100)
    hand_text = "Hand detected" if hand_detected else "No hand detected"
    cv2.putText(display, hand_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)

    buf_text = f"Buffer: {len(frame_buffer)} frames"
    cv2.putText(display, buf_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    conf_color = (0, 255, 0) if last_confidence >= 70 else (0, 165, 255) if last_confidence >= 40 else (0, 0, 255)
    pred_text = f"PREDICTED: {last_prediction}  ({last_confidence:.1f}%)"
    cv2.putText(display, pred_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, conf_color, 2)

    if last_top3:
        top3_text = f"Top3: {last_top3[0][0]}({last_top3[0][1]}%)  {last_top3[1][0]}({last_top3[1][1]}%)  {last_top3[2][0]}({last_top3[2][1]}%)"
        cv2.putText(display, top3_text, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    timer_remaining = max(0, PREDICT_EVERY - (now - last_pred_time))
    bar_width = int((1 - timer_remaining/PREDICT_EVERY) * w)
    cv2.rectangle(display, (0, h-5), (bar_width, h), (110, 231, 247), -1)

    instruction_y = h - 20
    cv2.putText(display, "SPACE=predict  C=clear  S=summary  Q=quit", (10, instruction_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    cv2.imshow("ISL LSTM Debug — SignBridge", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        if len(frame_buffer) >= 5:
            sequence = list(frame_buffer)
            pred, conf, top3 = predict_from_sequence(sequence)
            if pred:
                last_prediction = pred
                last_confidence = conf
                last_top3 = top3
                last_pred_time = time.time()
                prediction_count[pred] = prediction_count.get(pred, 0) + 1
                prediction_history.append((pred, conf))
                print(f"MANUAL PREDICT: {pred:<20} Confidence: {conf:.1f}%  |  Top3: {top3}")
    elif key == ord('c'):
        frame_buffer.clear()
        print("Buffer cleared.")
    elif key == ord('s'):
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        total_preds = sum(prediction_count.values())
        print(f"Total predictions made: {total_preds}")
        print("\nPrediction frequency per class:")
        sorted_counts = sorted(prediction_count.items(), key=lambda x: -x[1])
        for cls, count in sorted_counts:
            if count > 0:
                pct = count/total_preds*100 if total_preds > 0 else 0
                bar = "█" * int(pct/2)
                print(f"  {cls:<20} {count:3d} times ({pct:.1f}%) {bar}")
        never_predicted = [cls for cls, count in prediction_count.items() if count == 0]
        if never_predicted:
            print(f"\nNEVER predicted: {never_predicted}")
        print("="*60 + "\n")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("FINAL SESSION SUMMARY")
print("="*60)
total_preds = sum(prediction_count.values())
print(f"Total predictions: {total_preds}")
sorted_counts = sorted(prediction_count.items(), key=lambda x: -x[1])
print("\nAll class prediction counts:")
for cls, count in sorted_counts:
    pct = count/total_preds*100 if total_preds > 0 else 0
    print(f"  {cls:<20} {count:3d} times ({pct:.1f}%)")
never_predicted = [cls for cls, count in prediction_count.items() if count == 0]
if never_predicted:
    print(f"\nClasses NEVER predicted during session: {never_predicted}")
print("\nDebug session complete.")
