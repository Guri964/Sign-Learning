import os
import json
import numpy as np
import tensorflow as tf
from config import Config
from services import mlp_service
from utils.helpers import (
    extract_kinematic_features,
    normalize_wrist_relative,
    pad_or_truncate,
    stabilize_hand_sequence,
    standardize_sequence,
)

lstm_model = None
kinematic_model = None
lstm_mean = None
lstm_std = None
kin_mean = None
kin_std = None
label_map = None
label_classes = None
_proto_tensor = None
_proto_present = None
_compiled_forward = None

MIN_ACTIVE_FRAMES = 3
MAX_OOD_PROTO_DIST = 0.65
UNKNOWN_CLASS_IDX = 70


def load_lstm_model():
    global lstm_model, kinematic_model, lstm_mean, lstm_std, kin_mean, kin_std
    global label_map, label_classes, _proto_tensor, _proto_present, _compiled_forward

    if lstm_model is None:
        print(f"Loading 70-class studio LSTM backbone from {Config.LSTM_MODEL_PATH}...")
        lstm_model = tf.keras.models.load_model(Config.LSTM_MODEL_PATH, compile=False)
        lstm_mean = np.load(Config.LSTM_MEAN_PATH)
        lstm_std = np.load(Config.LSTM_STD_PATH)
        with open(Config.LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        label_classes = np.load(Config.LABEL_CLASSES_PATH, allow_pickle=True).tolist()
        print(f"LSTM backbone loaded with {len(label_classes)} classes.")

    if kinematic_model is None and os.path.exists(Config.WEBCAM_KINEMATIC_MODEL_PATH):
        print(f"Loading 71-class Open-Set Webcam Kinematic model from {Config.WEBCAM_KINEMATIC_MODEL_PATH}...")
        kinematic_model = tf.keras.models.load_model(
            Config.WEBCAM_KINEMATIC_MODEL_PATH,
            compile=False,
        )
        kin_mean = np.load(Config.KINEMATIC_MEAN_PATH)
        kin_std = np.load(Config.KINEMATIC_STD_PATH)
        print("Webcam Kinematic 71-class model loaded successfully.")

    if mlp_service.mlp_model is None:
        mlp_service.load_mlp_model()

    if _compiled_forward is None and kinematic_model is not None:
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[None, Config.SEQUENCE_LENGTH, 279], dtype=tf.float32),
                tf.TensorSpec(shape=[None, Config.SEQUENCE_LENGTH, Config.NUM_FEATURES], dtype=tf.float32),
                tf.TensorSpec(shape=[None, Config.SEQUENCE_LENGTH, Config.NUM_FEATURES], dtype=tf.float32),
            ]
        )
        def _run_ensemble(k_in, l_in, m_in):
            return (
                kinematic_model(k_in, training=False),
                lstm_model(l_in, training=False),
                mlp_service.mlp_model(m_in, training=False),
            )

        # Warm up TensorFlow graph once at startup (~12ms/request thereafter vs ~250ms eager)
        _run_ensemble(
            tf.zeros([1, Config.SEQUENCE_LENGTH, 279], dtype=tf.float32),
            tf.zeros([1, Config.SEQUENCE_LENGTH, Config.NUM_FEATURES], dtype=tf.float32),
            tf.zeros([1, Config.SEQUENCE_LENGTH, Config.NUM_FEATURES], dtype=tf.float32),
        )
        _compiled_forward = _run_ensemble

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


def mirror_raw_bimanual_sequence(seq_arr: np.ndarray) -> np.ndarray:
    """
    Mirrors a raw/wrist-centered (T, 126) bimanual sequence across horizontal image center x=0.5
    (or x=0 for pre-centered coordinates) and swaps Left <-> Right slots so left-handed signers
    and selfie-mirrored camera streams match right-dominant ISL sign chirality.
    """
    mir = np.zeros_like(seq_arr)
    lh = seq_arr[:, :63].reshape(len(seq_arr), 21, 3).copy()
    rh = seq_arr[:, 63:].reshape(len(seq_arr), 21, 3).copy()
    ml = np.any(lh != 0.0, axis=(1, 2))
    mr = np.any(rh != 0.0, axis=(1, 2))
    is_raw = (
        (np.any(ml) and float(np.max(np.linalg.norm(lh[ml, 0, :2], axis=-1))) > 1e-3)
        or (np.any(mr) and float(np.max(np.linalg.norm(rh[mr, 0, :2], axis=-1))) > 1e-3)
    )
    if np.any(ml):
        lh[ml, :, 0] = (1.0 - lh[ml, :, 0]) if is_raw else -lh[ml, :, 0]
        mir[:, 63:] = lh.reshape(len(seq_arr), 63)
    if np.any(mr):
        rh[mr, :, 0] = (1.0 - rh[mr, :, 0]) if is_raw else -rh[mr, :, 0]
        mir[:, :63] = rh.reshape(len(seq_arr), 63)
    return mir


def _compute_prototype_distances(sample_frames: np.ndarray) -> np.ndarray:
    k_len = sample_frames.shape[0]
    if k_len == 0 or _proto_tensor is None:
        return np.full(len(label_classes), np.inf, dtype=np.float32)

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


FINGER_CHAINS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))


def _has_valid_human_hand_anatomy(active_frames: np.ndarray) -> bool:
    """
    Verifies basic human phalanx joint geometry: consecutive phalanges (MCP->PIP->DIP->TIP)
    articulate forward along each finger (positive mean cosine across fingers), whereas
    uncorrelated random/corrupted point clouds satisfy E[(X_c - X_b).(X_b - X_a)] < 0 (~ -0.5).
    """
    cosines = []
    for base in (0, 63):
        hand = active_frames[:, base:base + 63].reshape(-1, 21, 3)
        present = np.any(hand != 0.0, axis=(1, 2))
        if np.any(present):
            sub = hand[present]
            for a, b, c, d in FINGER_CHAINS:
                v1 = sub[:, b] - sub[:, a]
                v2 = sub[:, c] - sub[:, b]
                v3 = sub[:, d] - sub[:, c]
                for u, v in ((v1, v2), (v2, v3)):
                    denom = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-8
                    cosines.append(float(np.mean(np.sum(u * v, axis=-1) / denom)))
    return bool(cosines) and float(np.mean(cosines)) > -0.35


def predict_word(sequence):
    """
    Predicts a 70-class ISL sign word from a sequence of 126-d landmark frames.
    Pipeline:
      1. Strict empty / low-frame / collapsed-coordinate rejection -> ("Unknown", 0.0, []).
      2. Temporal hand-slot stabilization (`stabilize_hand_sequence`) eliminating duplicate
         boxes on the same wrist, mid-clip single-hand label flips, and left-hand slot collapse.
      3. Physical hand-anatomy sanity bound (`min_proto_dist > 0.85` -> "Unknown").
      4. Scale- and position-invariant 279-d kinematic encoding (`extract_kinematic_features`)
         evaluated in one graph-compiled forward pass across canonical and left-dominant bimanual
         chiralities, fused via calibrated convex mixing with `isl_lstm_model_70` and
         `isl_mlp_model_70_handaug`.
      5. Open-Set rejection gate returning ("Unknown", 0.0, []) on out-of-vocabulary gestures.
    """
    if lstm_model is None or kinematic_model is None:
        load_lstm_model()

    if sequence is None or len(sequence) < MIN_ACTIVE_FRAMES:
        return "Unknown", 0.0, []

    arr_seq = np.asarray(sequence, dtype=np.float32)
    if arr_seq.ndim != 2 or arr_seq.shape[1] != Config.NUM_FEATURES:
        return "Unknown", 0.0, []
    arr_seq = np.nan_to_num(arr_seq, nan=0.0, posinf=0.0, neginf=0.0)

    clean_seq, active_indices = stabilize_hand_sequence(arr_seq)
    if len(active_indices) < MIN_ACTIVE_FRAMES:
        return "Unknown", 0.0, []

    wrist_rel = np.array([normalize_wrist_relative(frame) for frame in clean_seq], dtype=np.float32)
    active_frames = wrist_rel[active_indices]

    if int(np.count_nonzero(active_frames)) < MIN_ACTIVE_FRAMES * 6:
        return "Unknown", 0.0, []

    if not _has_valid_human_hand_anatomy(active_frames):
        return "Unknown", 0.0, []

    mid_start = max(0, len(active_frames) // 4)
    mid_end = max(mid_start + 1, (3 * len(active_frames)) // 4)
    step = max(1, (mid_end - mid_start) // 5)
    sample_frames = active_frames[mid_start:mid_end:step]

    dists = _compute_prototype_distances(sample_frames)
    min_dist = float(np.min(dists))
    if not np.isfinite(min_dist) or min_dist > MAX_OOD_PROTO_DIST:
        return "Unknown", 0.0, []

    num_cls = len(label_classes)
    candidates = [clean_seq]
    if np.any(clean_seq[:, :63] != 0.0) and np.any(clean_seq[:, 63:] != 0.0):
        candidates.append(mirror_raw_bimanual_sequence(clean_seq))

    l_batch = np.concatenate(
        [
            standardize_sequence(
                pad_or_truncate([normalize_wrist_relative(f) for f in c_seq], Config.SEQUENCE_LENGTH),
                lstm_mean,
                lstm_std,
            )
            for c_seq in candidates
        ],
        axis=0,
    ).astype(np.float32)
    m_batch = np.concatenate(
        [
            standardize_sequence(
                pad_or_truncate([normalize_wrist_relative(f) for f in c_seq], Config.SEQUENCE_LENGTH),
                mlp_service.mlp_mean,
                mlp_service.mlp_std,
            )
            for c_seq in candidates
        ],
        axis=0,
    ).astype(np.float32)

    if kinematic_model is not None and kin_mean is not None and kin_std is not None:
        k_batch = np.array(
            [
                (extract_kinematic_features(c_seq, length=Config.SEQUENCE_LENGTH, stabilize=False) - kin_mean) / kin_std
                for c_seq in candidates
            ],
            dtype=np.float32,
        )
        p71_b, pl_b, pm_b = [t.numpy() for t in _compiled_forward(k_batch, l_batch, m_batch)]
        best = None
        for bi in range(len(candidates)):
            p_ood_i = float(p71_b[bi, UNKNOWN_CLASS_IDX])
            pk_i = p71_b[bi, :num_cls] / max(1.0 - p_ood_i, 1e-6)
            pe_i = 0.72 * pk_i + 0.16 * pl_b[bi] + 0.12 * pm_b[bi]
            pe_i /= max(float(np.sum(pe_i)), 1e-8)
            cal_i = pe_i * (1.0 - 0.55 * p_ood_i)
            # Slight tie-break preference for canonical right-dominant orientation unless mirrored clearly wins
            rank_score = float(np.max(cal_i)) * (0.94 if bi == 1 else 1.0)
            if best is None or rank_score > best[0]:
                best = (rank_score, float(np.max(cal_i)), p_ood_i, pe_i, cal_i)
        _, max_cal, p_ood, p_ens, cal_probs = best
    else:
        pl_b = lstm_model(l_batch, training=False).numpy()
        pm_b = mlp_service.mlp_model(m_batch, training=False).numpy()
        p_ood = 0.0
        p_ens = 0.60 * pl_b[0] + 0.40 * pm_b[0]
        p_ens /= max(float(np.sum(p_ens)), 1e-8)
        cal_probs = p_ens
        max_cal = float(np.max(cal_probs))

    pred_idx = int(np.argmax(cal_probs))
    sorted_cal = np.sort(cal_probs)[::-1]
    margin = float(sorted_cal[0] - sorted_cal[1]) if len(sorted_cal) > 1 else max_cal
    entropy = float(-np.sum(p_ens * np.log(np.clip(p_ens, 1e-8, 1.0))) / np.log(num_cls))

    if (
        p_ood >= 0.44
        or max_cal < 0.38
        or (p_ood >= 0.22 and max_cal < 0.44)
        or (entropy > 0.56 and margin < 0.17)
    ):
        return "Unknown", 0.0, []

    confidence = round(max_cal * 100.0, 2)
    predicted_word = label_map.get(
        str(pred_idx),
        label_classes[pred_idx] if pred_idx < num_cls else "Unknown",
    )

    top3_idx = np.argsort(cal_probs)[::-1][:3]
    top3 = [
        {
            "word": label_map.get(str(idx), label_classes[idx] if idx < num_cls else f"Class_{idx}"),
            "confidence": round(float(cal_probs[idx]) * 100.0, 2),
        }
        for idx in top3_idx
    ]

    return predicted_word, confidence, top3
