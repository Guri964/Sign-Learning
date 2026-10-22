import os
import json
import numpy as np
import tensorflow as tf
from config import Config
from services import mlp_service
from utils.helpers import normalize_wrist_relative, pad_or_truncate, standardize_sequence

lstm_model = None
lstm_mean = None
lstm_std = None
label_map = None
label_classes = None
_proto_tensor = None
_proto_present = None

MIN_ACTIVE_FRAMES = 3
MAX_OOD_PROTO_DIST = 0.60


def load_lstm_model():
    global lstm_model, lstm_mean, lstm_std, label_map, label_classes, _proto_tensor, _proto_present
    if lstm_model is None:
        print(f"Loading 70-class LSTM model from {Config.LSTM_MODEL_PATH}...")
        lstm_model = tf.keras.models.load_model(Config.LSTM_MODEL_PATH)
        lstm_mean = np.load(Config.LSTM_MEAN_PATH)
        lstm_std = np.load(Config.LSTM_STD_PATH)
        with open(Config.LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)
        label_classes = np.load(Config.LABEL_CLASSES_PATH, allow_pickle=True).tolist()
        print(f"LSTM model loaded successfully with {len(label_classes)} classes.")

    if mlp_service.mlp_model is None:
        mlp_service.load_mlp_model()

    if _proto_tensor is None:
        num_cls = len(label_classes)
        _proto_tensor = np.zeros((num_cls, 2, 21, 3), dtype=np.float32)
        _proto_present = np.zeros((num_cls, 2), dtype=bool)
        shapes = mlp_service.reference_shapes or {}
        for ci, cls_name in enumerate(label_classes):
            sh = shapes.get(cls_name, {})
            if sh.get("left") is not None:
                _proto_tensor[ci, 0] = np.asarray(sh["left"], dtype=np.float32)
                _proto_present[ci, 0] = True
            if sh.get("right") is not None:
                _proto_tensor[ci, 1] = np.asarray(sh["right"], dtype=np.float32)
                _proto_present[ci, 1] = True


def swap_and_mirror_chirality(seq_20x126: np.ndarray) -> np.ndarray:
    """
    True sagittal reflection across camera x-axis:
    swaps Left <-> Right hand slots AND negates wrist-relative x (-x),
    preserving human thumb/pinky chirality instead of creating OOD inputs.
    """
    out = np.zeros_like(seq_20x126)
    left = seq_20x126[:, :63].reshape(-1, 21, 3).copy()
    right = seq_20x126[:, 63:].reshape(-1, 21, 3).copy()
    left[:, :, 0] *= -1.0
    right[:, :, 0] *= -1.0
    out[:, :63] = right.reshape(-1, 63)
    out[:, 63:] = left.reshape(-1, 63)
    return out


def _compute_prototype_distances(sample_frames: np.ndarray) -> np.ndarray:
    """
    Vectorized tolerant L2 distance between user active hand frames and all 70
    train-set median hand prototypes (`reference_shapes_70.json`).
    """
    k_len = len(sample_frames)
    num_cls = len(label_classes)
    u = sample_frames.reshape(k_len, 2, 21, 3)
    up = np.count_nonzero(u.reshape(k_len, 2, 63), axis=2) > 0
    u_flip = u.copy()
    u_flip[:, :, :, 0] *= -1.0

    d_raw = np.mean(np.linalg.norm(u[:, :, None, None, :, :] - _proto_tensor[None, None, :, :, :, :], axis=-1), axis=-1)
    d_flp = np.mean(np.linalg.norm(u_flip[:, :, None, None, :, :] - _proto_tensor[None, None, :, :, :, :], axis=-1), axis=-1)
    d_hand = np.minimum(d_raw, d_flp)

    frame_cls_dist = np.full((k_len, num_cls), np.inf, dtype=np.float32)
    for k in range(k_len):
        for pr0, pr1 in ((0, 1), (1, 0)):
            num = np.zeros(num_cls, dtype=np.float32)
            cnt = np.zeros(num_cls, dtype=np.float32)
            for ui, ri in ((0, pr0), (1, pr1)):
                if up[k, ui]:
                    mask = _proto_present[:, ri]
                    num[mask] += d_hand[k, ui, mask, ri]
                    cnt[mask] += 1.0
            ok = cnt > 0
            cand = np.where(ok, num / np.maximum(cnt, 1.0), np.inf)
            frame_cls_dist[k] = np.minimum(frame_cls_dist[k], cand)

    return np.mean(frame_cls_dist, axis=0)


def predict_word(sequence):
    """
    Predicts a 70-class ISL sign word from a sequence of 126-d landmark frames.
    Fixes:
      1. Empty / low-frame / non-hand OOD rejection -> ("Unknown", 0.0, [])
      2. Anatomically valid sagittal mirror-swap (x -> -x with slot swap)
      3. INCLUDE temporal active-window alignment for continuous webcam captures
      4. Adaptive log-posterior fusion combining 70-class temporal LSTM,
         70-class hand-augmented MLP (single-hand resilient), and train-set hand shapes.
    """
    if lstm_model is None:
        load_lstm_model()

    if not sequence or len(sequence) < MIN_ACTIVE_FRAMES:
        return "Unknown", 0.0, []

    arr_seq = np.asarray(sequence, dtype=np.float32)
    if arr_seq.ndim != 2 or arr_seq.shape[1] != Config.NUM_FEATURES:
        return "Unknown", 0.0, []
    arr_seq = np.nan_to_num(arr_seq, nan=0.0, posinf=0.0, neginf=0.0)

    active_mask = np.count_nonzero(arr_seq, axis=1) >= 6
    if int(np.sum(active_mask)) < MIN_ACTIVE_FRAMES:
        return "Unknown", 0.0, []

    wrist_rel = np.array([normalize_wrist_relative(frame) for frame in arr_seq], dtype=np.float32)
    active_frames = wrist_rel[active_mask]

    # Reject degenerate constant-coordinate inputs (e.g., all 21 landmarks identical -> zero after wrist subtraction)
    if int(np.count_nonzero(active_frames)) < MIN_ACTIVE_FRAMES * 6:
        return "Unknown", 0.0, []

    mid_start = max(0, len(active_frames) // 4)
    mid_end = max(mid_start + 1, (3 * len(active_frames)) // 4)
    step = max(1, (mid_end - mid_start) // 5)
    sample_frames = active_frames[mid_start:mid_end:step]

    dists = _compute_prototype_distances(sample_frames)
    min_dist = float(np.min(dists))
    if not np.isfinite(min_dist) or min_dist > MAX_OOD_PROTO_DIST:
        return "Unknown", 0.0, []

    padded = pad_or_truncate(wrist_rel, Config.SEQUENCE_LENGTH)

    # Align sustained webcam windows (where hand stays active across all 20 frames) with INCLUDE clip active envelope
    enveloped = padded.copy()
    if float(np.mean(np.count_nonzero(padded, axis=1) > 0)) >= 0.85:
        enveloped[:3] = 0.0
        enveloped[17:] = 0.0

    lstm_batch = np.concatenate([
        standardize_sequence(padded, lstm_mean, lstm_std),
        standardize_sequence(swap_and_mirror_chirality(padded), lstm_mean, lstm_std),
        standardize_sequence(enveloped, lstm_mean, lstm_std),
        standardize_sequence(swap_and_mirror_chirality(enveloped), lstm_mean, lstm_std),
    ], axis=0)

    mlp_batch = np.concatenate([
        standardize_sequence(enveloped, mlp_service.mlp_mean, mlp_service.mlp_std),
        standardize_sequence(swap_and_mirror_chirality(enveloped), mlp_service.mlp_mean, mlp_service.mlp_std),
    ], axis=0)

    lstm_probs = lstm_model(tf.constant(lstm_batch, dtype=tf.float32), training=False).numpy()
    mlp_probs = mlp_service.mlp_model(tf.constant(mlp_batch, dtype=tf.float32), training=False).numpy()

    p_lstm = np.max(lstm_probs, axis=0)
    p_mlp = np.max(mlp_probs, axis=0)
    finite_d = np.where(np.isfinite(dists), dists, 2.0)

    both_hands_ratio = float(np.mean(
        (np.count_nonzero(active_frames[:, :63], axis=1) > 0) &
        (np.count_nonzero(active_frames[:, 63:], axis=1) > 0)
    ))
    if len(active_frames) >= 6:
        early_mean = np.mean(active_frames[:len(active_frames) // 3], axis=0)
        late_mean = np.mean(active_frames[-len(active_frames) // 3:], axis=0)
        net_displacement = float(np.linalg.norm(late_mean - early_mean))
    else:
        net_displacement = 0.0

    if both_hands_ratio >= 0.5 and net_displacement > 0.55:
        # True multi-pose dynamic motion trajectory: favor temporal LSTM + hand-augmented MLP
        log_post = 0.55 * np.log(p_lstm + 1e-4) + 0.30 * np.log(p_mlp + 1e-4) - 4.0 * finite_d
    else:
        # Sustained live webcam pose or single-hand signer: combine neural likelihoods + train-only prototype geometry
        log_post = 0.35 * np.log(p_lstm + 1e-4) + 0.25 * np.log(p_mlp + 1e-4) - 14.0 * finite_d

    log_post -= np.max(log_post)
    post = np.exp(log_post) / (np.sum(np.exp(log_post)) + 1e-12)

    pred_idx = int(np.argmax(post))
    confidence = float(post[pred_idx]) * 100.0

    predicted_word = label_map.get(
        str(pred_idx),
        label_classes[pred_idx] if pred_idx < len(label_classes) else "Unknown",
    )

    top3_idx = np.argsort(post)[::-1][:3]
    top3 = [
        {
            "word": label_map.get(str(idx), label_classes[idx] if idx < len(label_classes) else f"Class_{idx}"),
            "confidence": round(float(post[idx]) * 100.0, 2),
        }
        for idx in top3_idx
    ]

    return predicted_word, round(confidence, 2), top3
