import json
import numpy as np
import tensorflow as tf
from config import Config
from utils.helpers import normalize_wrist_relative, pad_or_truncate, standardize_sequence

lstm_model = None
lstm_mean = None
lstm_std = None
label_map = None
label_classes = None

def load_lstm_model():
    global lstm_model, lstm_mean, lstm_std, label_map, label_classes
    if lstm_model is None:
        print(f"Loading 70-class LSTM model from {Config.LSTM_MODEL_PATH}...")
        lstm_model = tf.keras.models.load_model(Config.LSTM_MODEL_PATH)
        lstm_mean = np.load(Config.LSTM_MEAN_PATH)
        lstm_std = np.load(Config.LSTM_STD_PATH)
        with open(Config.LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)
        label_classes = np.load(Config.LABEL_CLASSES_PATH, allow_pickle=True).tolist()
        print(f"LSTM model loaded successfully with {len(label_classes)} classes.")

def predict_word(sequence):
    """
    Predicts sign word from a sequence of 126-d landmark frames.
    Preprocessing matches 100% of the training pipeline:
      1. Wrist-relative normalization per frame
      2. Linspace temporal resampling / padding to 20 frames
      3. Global Z-score standardization: (X - mean) / (std + 1e-8)
      4. Dual slot evaluation (direct + swapped) to accommodate mirror webcam
         view and single-handed / left-handed execution.
    """
    if lstm_model is None:
        load_lstm_model()

    if not sequence or len(sequence) == 0:
        return "Unknown", 0.0, []

    raw_seq = [np.array(f, dtype=np.float32) for f in sequence]

    # Filter frames: if user has any hand frames, use only non-empty frames for cleaner resampling
    non_empty = [f for f in raw_seq if np.count_nonzero(f) > 0]
    frames_to_use = non_empty if len(non_empty) >= 5 else raw_seq

    # Step 1: Wrist-relative normalization per frame
    wrist_rel_seq = [normalize_wrist_relative(lm) for lm in frames_to_use]

    # Step 2: Linspace temporal resampling / padding to 20 frames
    padded = pad_or_truncate(wrist_rel_seq, Config.SEQUENCE_LENGTH)

    # Step 3: Global Z-Score standardization (direct slot)
    normalized = standardize_sequence(padded, lstm_mean, lstm_std)
    input_data = normalized.reshape(1, Config.SEQUENCE_LENGTH, Config.NUM_FEATURES)

    # Step 4: Model Inference - Direct slot
    probs_direct = lstm_model.predict(input_data, verbose=0)[0]

    # Step 5: Model Inference - Swapped slot (Left <-> Right)
    # Essential for single-hand signs signed with the user's non-dominant or mirrored hand
    perm = np.r_[63:126, 0:63]
    padded_swapped = padded[:, perm]
    normalized_swapped = standardize_sequence(padded_swapped, lstm_mean, lstm_std).reshape(1, Config.SEQUENCE_LENGTH, Config.NUM_FEATURES)
    probs_swapped = lstm_model.predict(normalized_swapped, verbose=0)[0]

    max_direct = float(np.max(probs_direct))
    max_swapped = float(np.max(probs_swapped))

    if max_swapped > max_direct:
        probs = probs_swapped
    else:
        probs = probs_direct

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx]) * 100.0

    predicted_word = label_map.get(str(pred_idx), label_classes[pred_idx] if pred_idx < len(label_classes) else "Unknown")

    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = []
    for idx in top3_idx:
        word = label_map.get(str(idx), label_classes[idx] if idx < len(label_classes) else f"Class_{idx}")
        top3.append({
            "word": word,
            "confidence": round(float(probs[idx]) * 100.0, 2)
        })

    return predicted_word, round(confidence, 2), top3