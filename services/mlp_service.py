import os
import json
import numpy as np
import tensorflow as tf
from config import Config
from utils.helpers import (
    normalize_wrist_relative,
    pad_or_truncate,
    standardize_sequence,
    calculate_shape_accuracy,
    get_finger_feedback
)

mlp_model = None
mlp_mean = None
mlp_std = None
reference_shapes = None
reference_landmarks = None
label_classes = None

def load_mlp_model():
    global mlp_model, mlp_mean, mlp_std, reference_shapes, reference_landmarks, label_classes
    if mlp_model is None:
        print(f"Loading 70-class Hand-Augmented MLP model from {Config.MLP_MODEL_PATH}...")
        mlp_model = tf.keras.models.load_model(Config.MLP_MODEL_PATH)
        mlp_mean = np.load(Config.MLP_MEAN_PATH)
        mlp_std = np.load(Config.MLP_STD_PATH)
        
        if os.path.exists(Config.REFERENCE_SHAPES_PATH):
            with open(Config.REFERENCE_SHAPES_PATH, "r") as f:
                reference_shapes = json.load(f)
        else:
            reference_shapes = {}

        if os.path.exists(Config.REFERENCE_LANDMARKS_PATH):
            with open(Config.REFERENCE_LANDMARKS_PATH, "r") as f:
                reference_landmarks = json.load(f)
        else:
            reference_landmarks = {}

        label_classes = np.load(Config.LABEL_CLASSES_PATH, allow_pickle=True).tolist()
        print(f"MLP model loaded successfully with {len(label_classes)} classes and {len(reference_shapes)} reference shapes.")

def predict_and_score(landmarks_input, selected_word):
    """
    Evaluates an attempt against a selected target word.
    Accepts either:
      - a single 126-d landmark array
      - a sequence of 126-d frames (e.g. 1.5 - 2s of frames)
    Returns:
      predicted_word, confidence, accuracy (0-100), feedback, badge_unlocked
    """
    if mlp_model is None:
        load_mlp_model()

    # Determine if landmarks_input is a single frame or a sequence
    is_sequence = isinstance(landmarks_input, (list, np.ndarray)) and len(landmarks_input) > 0 and isinstance(landmarks_input[0], (list, np.ndarray))

    if is_sequence:
        seq_raw = [np.array(f, dtype=np.float32) for f in landmarks_input]
        wrist_rel_seq = [normalize_wrist_relative(f) for f in seq_raw]
        # Middle frame for shape accuracy
        mid_idx = len(wrist_rel_seq) // 2
        mid_frame = wrist_rel_seq[mid_idx]
    else:
        frame_raw = np.array(landmarks_input, dtype=np.float32).flatten()
        mid_frame = normalize_wrist_relative(frame_raw)
        wrist_rel_seq = [mid_frame] * Config.SEQUENCE_LENGTH

    # 1. Shape comparison with reference_shapes_70.json
    ref_shape = reference_shapes.get(selected_word) if reference_shapes else None
    shape_accuracy = 0.0
    feedback = "Perform the sign clearly facing the camera."
    
    if ref_shape is not None:
        shape_accuracy, dist = calculate_shape_accuracy(mid_frame, ref_shape)
        feedback = get_finger_feedback(mid_frame, ref_shape)
    elif reference_landmarks and selected_word in reference_landmarks:
        # Fallback to reference_landmarks_70
        ref_raw = np.array(reference_landmarks[selected_word], dtype=np.float32)
        ref_rel = normalize_wrist_relative(ref_raw)
        ref_dummy_shape = {
            "left": ref_rel[:63].reshape(21, 3).tolist() if np.count_nonzero(ref_rel[:63]) > 0 else None,
            "right": ref_rel[63:].reshape(21, 3).tolist() if np.count_nonzero(ref_rel[63:]) > 0 else None
        }
        shape_accuracy, dist = calculate_shape_accuracy(mid_frame, ref_dummy_shape)
        feedback = get_finger_feedback(mid_frame, ref_dummy_shape)

    # 2. Model prediction
    padded = pad_or_truncate(wrist_rel_seq, Config.SEQUENCE_LENGTH)
    normalized = standardize_sequence(padded, mlp_mean, mlp_std)
    input_data = normalized.reshape(1, Config.SEQUENCE_LENGTH, Config.NUM_FEATURES)

    probs = mlp_model.predict(input_data, verbose=0)[0]

    # Dual-slot check: evaluate with swapped slots (Left <-> Right) to handle mirror camera
    perm = np.r_[63:126, 0:63]
    padded_swapped = padded[:, perm]
    normalized_swapped = standardize_sequence(padded_swapped, mlp_mean, mlp_std).reshape(1, Config.SEQUENCE_LENGTH, Config.NUM_FEATURES)
    probs_swapped = mlp_model.predict(normalized_swapped, verbose=0)[0]

    target_idx = label_classes.index(selected_word) if selected_word in label_classes else -1
    target_prob_direct = float(probs[target_idx]) * 100.0 if target_idx >= 0 else 0.0
    target_prob_swapped = float(probs_swapped[target_idx]) * 100.0 if target_idx >= 0 else 0.0

    if target_prob_swapped > target_prob_direct:
        probs = probs_swapped
        target_prob = target_prob_swapped
    else:
        target_prob = target_prob_direct

    pred_idx = int(np.argmax(probs))
    predicted_word = label_classes[pred_idx] if pred_idx < len(label_classes) else "Unknown"
    confidence = float(probs[pred_idx]) * 100.0

    # Combined composite accuracy (lenient scoring):
    # Take the BEST signal: geometric shape OR neural confidence for the target word
    # This prevents a bad shape score from burying a good model prediction and vice-versa
    if target_prob > 0.0 and shape_accuracy > 0.0:
        # Weighted blend, leaning toward whichever signal is stronger
        composite_acc = max(
            (shape_accuracy * 0.4) + (target_prob * 0.6),
            (shape_accuracy * 0.6) + (target_prob * 0.4)
        )
    elif target_prob > 0.0:
        composite_acc = target_prob
    else:
        composite_acc = shape_accuracy

    # Boost generously if model top prediction matches the target
    if predicted_word == selected_word:
        composite_acc = max(composite_acc, confidence * 0.90)

    # Give a floor: if model has at least some confidence for target (>5%), show at least 15%
    if target_prob > 5.0:
        composite_acc = max(composite_acc, 15.0)

    final_accuracy = round(float(np.clip(composite_acc, 0.0, 100.0)), 2)

    return {
        "predicted_word": predicted_word,
        "confidence": round(confidence, 2),
        "target_prob": round(target_prob, 2),
        "accuracy": final_accuracy,
        "feedback": feedback,
        "badge_unlocked": final_accuracy >= Config.BADGE_THRESHOLD
    }

def get_reference_landmarks(word):
    if reference_landmarks is None:
        load_mlp_model()
    return reference_landmarks.get(word, None)

def get_reference_shape(word):
    if reference_shapes is None:
        load_mlp_model()
    return reference_shapes.get(word, None)

def get_all_words():
    if label_classes is None:
        load_mlp_model()
    return label_classes