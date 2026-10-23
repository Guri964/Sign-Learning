import os
import json
import numpy as np
import tensorflow as tf
from config import Config
from utils.helpers import (
    normalize_wrist_relative,
    pad_or_truncate,
    stabilize_hand_sequence,
    standardize_sequence,
    calculate_shape_accuracy,
    get_finger_feedback,
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


def _mirror_swap_sequence(padded_20x126: np.ndarray) -> np.ndarray:
    """Swaps Left <-> Right slots with anatomical sagittal x-reflection (-x)."""
    out = np.zeros_like(padded_20x126)
    lh = padded_20x126[:, :63].reshape(-1, 21, 3).copy()
    rh = padded_20x126[:, 63:].reshape(-1, 21, 3).copy()
    lh[:, :, 0] *= -1.0
    rh[:, :, 0] *= -1.0
    out[:, :63] = rh.reshape(-1, 63)
    out[:, 63:] = lh.reshape(-1, 63)
    return out


def predict_and_score(landmarks_input, selected_word):
    """
    Evaluates an attempt against a selected target word in AI Tutor.
    Accepts either:
      - a single 126-d landmark array
      - a sequence of 126-d frames (e.g. 1.5 - 2s of frames)
    Returns:
      predicted_word, confidence, target_prob, accuracy (0-100), feedback, badge_unlocked
    """
    if mlp_model is None:
        load_mlp_model()

    is_sequence = (
        isinstance(landmarks_input, (list, np.ndarray))
        and len(landmarks_input) > 0
        and isinstance(landmarks_input[0], (list, np.ndarray))
    )

    if is_sequence:
        clean_seq, active_idx = stabilize_hand_sequence(landmarks_input)
        if not active_idx:
            return {
                "predicted_word": "Unknown",
                "confidence": 0.0,
                "target_prob": 0.0,
                "accuracy": 0.0,
                "feedback": "Show your hand clearly in front of the camera.",
                "badge_unlocked": False,
            }
        wrist_rel_seq = [normalize_wrist_relative(f) for f in clean_seq]
        mid_idx = active_idx[len(active_idx) // 2]
        mid_frame = wrist_rel_seq[mid_idx]
    else:
        frame_raw = np.asarray(landmarks_input, dtype=np.float32).flatten()
        clean_seq, active_idx = stabilize_hand_sequence(frame_raw.reshape(1, -1))
        if not active_idx:
            return {
                "predicted_word": "Unknown",
                "confidence": 0.0,
                "target_prob": 0.0,
                "accuracy": 0.0,
                "feedback": "Show your hand clearly in front of the camera.",
                "badge_unlocked": False,
            }
        mid_frame = normalize_wrist_relative(clean_seq[0])
        if np.count_nonzero(mid_frame) < 6:
            return {
                "predicted_word": "Unknown",
                "confidence": 0.0,
                "target_prob": 0.0,
                "accuracy": 0.0,
                "feedback": "Show your hand clearly in front of the camera.",
                "badge_unlocked": False,
            }
        wrist_rel_seq = [mid_frame] * Config.SEQUENCE_LENGTH

    # 1. Geometric hand-shape evaluation across peak stroke frames against target prototype
    ref_shape = reference_shapes.get(selected_word) if reference_shapes else None
    shape_accuracy = 0.0
    feedback = "Perform the sign clearly facing the camera."

    candidate_frames = [wrist_rel_seq[i] for i in active_idx] if is_sequence else [mid_frame]
    eval_shape_target = ref_shape
    if eval_shape_target is None and reference_landmarks and selected_word in reference_landmarks:
        ref_raw = np.array(reference_landmarks[selected_word], dtype=np.float32)
        ref_rel = normalize_wrist_relative(ref_raw)
        eval_shape_target = {
            "left": ref_rel[:63].reshape(21, 3).tolist() if np.count_nonzero(ref_rel[:63]) > 0 else None,
            "right": ref_rel[63:].reshape(21, 3).tolist() if np.count_nonzero(ref_rel[63:]) > 0 else None,
        }

    if eval_shape_target is not None:
        best_frame = mid_frame
        best_s_acc = -1.0
        for cf in candidate_frames:
            s_acc, _ = calculate_shape_accuracy(cf, eval_shape_target)
            if s_acc > best_s_acc:
                best_s_acc = s_acc
                best_frame = cf
        shape_accuracy = max(0.0, best_s_acc)
        feedback = get_finger_feedback(best_frame, eval_shape_target)

    # 2. Neural classification (anatomically valid mirror-swap + kinematic sequence check when >=3 frames)
    padded = pad_or_truncate(wrist_rel_seq, Config.SEQUENCE_LENGTH)
    padded_swapped = _mirror_swap_sequence(padded)
    mlp_batch = np.concatenate([
        standardize_sequence(padded, mlp_mean, mlp_std),
        standardize_sequence(padded_swapped, mlp_mean, mlp_std),
    ], axis=0)
    mlp_out = mlp_model(tf.constant(mlp_batch, dtype=tf.float32), training=False).numpy()
    probs_direct, probs_swapped = mlp_out[0], mlp_out[1]

    target_idx = label_classes.index(selected_word) if selected_word in label_classes else -1
    t_direct = float(probs_direct[target_idx]) * 100.0 if target_idx >= 0 else 0.0
    t_swapped = float(probs_swapped[target_idx]) * 100.0 if target_idx >= 0 else 0.0

    probs = probs_swapped if t_swapped > t_direct else probs_direct
    target_prob = max(t_direct, t_swapped)

    # If a genuine multi-frame sequence was captured, blend with the 71-class Kinematic+LSTM ensemble
    if is_sequence and len(active_idx) >= 3:
        from services import lstm_service
        seq_word, seq_conf, seq_top3 = lstm_service.predict_word(clean_seq)
        if seq_word != "Unknown":
            for item in seq_top3:
                if item["word"] == selected_word:
                    target_prob = max(target_prob, float(item["confidence"]))
            if seq_word == selected_word or seq_conf > float(np.max(probs)) * 100.0:
                pred_word_final = seq_word
                conf_final = seq_conf
            else:
                pred_idx = int(np.argmax(probs))
                pred_word_final = label_classes[pred_idx] if pred_idx < len(label_classes) else "Unknown"
                conf_final = float(probs[pred_idx]) * 100.0
        else:
            pred_idx = int(np.argmax(probs))
            pred_word_final = label_classes[pred_idx] if pred_idx < len(label_classes) else "Unknown"
            conf_final = float(probs[pred_idx]) * 100.0
    else:
        pred_idx = int(np.argmax(probs))
        pred_word_final = label_classes[pred_idx] if pred_idx < len(label_classes) else "Unknown"
        conf_final = float(probs[pred_idx]) * 100.0

    if target_prob > 0.0 and shape_accuracy > 0.0:
        composite_acc = max(
            (shape_accuracy * 0.4) + (target_prob * 0.6),
            (shape_accuracy * 0.6) + (target_prob * 0.4),
        )
    elif target_prob > 0.0:
        composite_acc = target_prob
    else:
        composite_acc = shape_accuracy

    if pred_word_final == selected_word:
        composite_acc = max(composite_acc, conf_final * 0.90)

    if target_prob > 5.0:
        composite_acc = max(composite_acc, 15.0)

    final_accuracy = round(float(np.clip(composite_acc, 0.0, 100.0)), 2)

    return {
        "predicted_word": pred_word_final,
        "confidence": round(float(conf_final), 2),
        "target_prob": round(float(target_prob), 2),
        "accuracy": final_accuracy,
        "feedback": feedback,
        "badge_unlocked": final_accuracy >= Config.BADGE_THRESHOLD,
    }


def get_reference_landmarks(word):
    if reference_landmarks is None:
        load_mlp_model()
    return reference_landmarks.get(word, None)


def get_reference_shape(word):
    if reference_shapes is None:
        load_mlp_model()
    return reference_shapes.get(word, None)


def get_target_ghost_hand(word):
    """Returns a single 21x3 wrist-relative right-hand target skeleton for webcam ghost-hand overlay."""
    if reference_shapes is None:
        load_mlp_model()
    shape = reference_shapes.get(word)
    if shape and shape.get("right") is not None:
        return shape["right"]
    raw = reference_landmarks.get(word)
    if raw is not None and len(raw) >= 126:
        right_arr = np.asarray(raw[63:126], dtype=np.float32)
        if np.any(np.abs(right_arr) > 1e-6):
            norm126 = normalize_wrist_relative(np.asarray(raw, dtype=np.float32))
            return norm126[63:126].reshape(21, 3).tolist()
    if shape and shape.get("left") is not None:
        return [[-pt[0], pt[1], pt[2]] for pt in shape["left"]]
    return None


def get_all_words():
    if label_classes is None:
        load_mlp_model()
    return label_classes
