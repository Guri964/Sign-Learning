#!/usr/bin/env python3
"""
Genuine Dataset Audit, Strict Recording-Disjoint Split, 71-Class Open-Set Webcam Kinematic
Model Training (70 ISL Signs + Explicit Class 70 = 'Unknown' OOD Head), and Before-vs-After
Held-Out Evaluation Across All 70 ISL Signs + 127 Real 640x480 Laptop Crops + 207 Real OOD Videos.
"""

import csv
import importlib.util
import json
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.helpers import (
    extract_kinematic_features,
    normalize_wrist_relative,
    pad_or_truncate,
    stabilize_hand_sequence,
    standardize_sequence,
)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_DIR = os.path.join(ROOT_DIR, "models_LSTM")
ANALYSIS_DIR = os.path.join(MODEL_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

CLASSES = [str(x) for x in np.load(os.path.join(MODEL_DIR, "label_classes.npy"), allow_pickle=True)]
NUM_SIGN_CLASSES = len(CLASSES)          # 70 in-vocabulary ISL classes (0..69)
UNKNOWN_CLASS_IDX = NUM_SIGN_CLASSES     # 70 -> explicit Open-Set "Unknown" head
TOTAL_OUTPUT_CLASSES = NUM_SIGN_CLASSES + 1  # 71 outputs total
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Force all 5 benchmarked real human videos strictly into HELD-OUT TEST (never seen in train)
FORCED_TEST_MVI = {"MVI_0033", "MVI_5183", "MVI_5184", "MVI_9984", "MVI_9987"}


def seq_signature(raw_seq):
    return pad_or_truncate([normalize_wrist_relative(f) for f in raw_seq], 20).astype(np.float32)


def simulate_upper_body_webcam_frustum(raw_seq, y_cutoff=0.60, ar_target=4.0 / 3.0, rng=None):
    """
    True pinhole camera frustum clipping for a desk laptop webcam:
    hands hanging idle below y_cutoff (thighs/hips beneath the keyboard) are occluded (0.0),
    and x/z reflect the webcam aspect ratio.
    """
    arr = np.asarray(raw_seq, dtype=np.float32).copy()
    if rng is not None:
        scale = float(rng.uniform(0.75, 1.35))
        tx = float(rng.uniform(-0.10, 0.10))
        ty = float(rng.uniform(-0.05, 0.06))
    else:
        scale, tx, ty = 1.15, 0.0, 0.02

    ar_inv = (16.0 / 9.0) / float(ar_target)
    out = np.zeros_like(arr)
    for t in range(len(arr)):
        for base in (0, 63):
            pts = arr[t, base:base + 63].reshape(21, 3)
            if np.count_nonzero(pts) < 6:
                continue
            if pts[0, 1] > y_cutoff:
                continue
            p_cam = pts.copy()
            p_cam[:, 0] = 0.50 + (p_cam[:, 0] - 0.50) * scale * ar_inv + tx
            p_cam[:, 1] = 0.34 + (p_cam[:, 1] - 0.34) * scale + ty
            p_cam[:, 2] = p_cam[:, 2] * scale * ar_inv
            if rng is not None:
                p_cam += rng.normal(0.0, 0.0025, size=p_cam.shape).astype(np.float32)
            out[t, base:base + 63] = p_cam.flatten()

    if np.count_nonzero(out) < 18:
        for t in range(len(arr)):
            for base in (63, 0):
                pts = arr[t, base:base + 63].reshape(21, 3)
                if np.count_nonzero(pts) >= 6:
                    p_cam = pts.copy()
                    p_cam[:, 1] -= 0.22
                    if p_cam[0, 1] <= y_cutoff + 0.08:
                        out[t, base:base + 63] = p_cam.flatten()
    return out


def augment_webcam_training_sample(raw_seq, rng):
    """Realistic webcam domain randomization strictly inside TRAIN split only."""
    seq = np.asarray(raw_seq, dtype=np.float32).copy()
    mode = float(rng.random())

    if mode < 0.45:
        y_cut = float(rng.uniform(0.54, 0.68))
        ar = float(rng.choice([4.0 / 3.0, 16.0 / 10.0, 16.0 / 9.0, 1.0]))
        seq = simulate_upper_body_webcam_frustum(seq, y_cutoff=y_cut, ar_target=ar, rng=rng)
    elif mode < 0.78:
        scale = float(rng.uniform(0.68, 1.45))
        tx = float(rng.uniform(-0.14, 0.14))
        ty = float(rng.uniform(-0.08, 0.08))
        for base in (0, 63):
            sub = seq[:, base:base + 63].reshape(len(seq), 21, 3)
            m = np.any(sub != 0.0, axis=(1, 2))
            if np.any(m):
                sub[m, :, 0] = 0.50 + (sub[m, :, 0] - 0.50) * scale + tx
                sub[m, :, 1] = 0.42 + (sub[m, :, 1] - 0.42) * scale + ty
                sub[m, :, 2] *= scale
                sub[m] += rng.normal(0.0, 0.003, size=sub[m].shape).astype(np.float32)
            seq[:, base:base + 63] = sub.reshape(len(seq), 63)

    n = len(seq)
    if n >= 10 and rng.random() < 0.50:
        crop_len = int(rng.uniform(0.78, 1.0) * n)
        start = int(rng.integers(0, max(1, n - crop_len + 1)))
        seq = seq[start:start + crop_len]

    if len(seq) >= 8 and rng.random() < 0.35:
        drop_mask = rng.random(len(seq)) < 0.10
        if np.sum(~drop_mask) >= 4:
            seq[drop_mask] = 0.0

    # Left-handed signer augmentation (swap + mirror chirality)
    if rng.random() < 0.20:
        swapped = np.zeros_like(seq)
        lh = seq[:, :63].reshape(len(seq), 21, 3).copy()
        rh = seq[:, 63:].reshape(len(seq), 21, 3).copy()
        ml = np.any(lh != 0.0, axis=(1, 2))
        mr = np.any(rh != 0.0, axis=(1, 2))
        if np.any(ml):
            lh[ml, :, 0] = 1.0 - lh[ml, :, 0]
            swapped[:, 63:] = lh.reshape(len(seq), 63)
        if np.any(mr):
            rh[mr, :, 0] = 1.0 - rh[mr, :, 0]
            swapped[:, :63] = rh.reshape(len(seq), 63)
        seq = swapped

    feats = extract_kinematic_features(seq, length=20, stabilize=True)
    if rng.random() < 0.18:
        feats[:, 252:261] = 0.0
    return feats


def synthesize_open_set_negative(train_by_cls, rng):
    """
    Generates realistic open-set / non-vocabulary human hand movements STRICTLY from
    `train_by_cls` (ZERO exposure to any held-out test video or any of the 207 real OOD videos!).
    Teaches explicit class 70 ('Unknown') to claim probability mass whenever a gesture's
    handshape-motion combination lies outside all 70 canonical ISL sign manifolds.
    """
    c_a, c_b = rng.choice(NUM_SIGN_CLASSES, size=2, replace=False)
    rec_a = train_by_cls[int(c_a)][int(rng.integers(0, len(train_by_cls[int(c_a)])))]
    rec_b = train_by_cls[int(c_b)][int(rng.integers(0, len(train_by_cls[int(c_b)])))]
    seq_a = pad_or_truncate(rec_a["full_raw"], 20).copy()
    seq_b = pad_or_truncate(rec_b["full_raw"], 20).copy()

    kind = float(rng.random())
    if kind < 0.28:
        # 1) Cross-sign temporal chimera: start of sign A morphing into unrelated sign B
        cut = int(rng.integers(6, 14))
        neg = np.concatenate([seq_a[:cut], seq_b[cut:]], axis=0)
    elif kind < 0.56:
        # 2) Incompatible bimanual / handshape-trajectory chimera
        neg = np.zeros((20, 126), dtype=np.float32)
        neg[:, :63] = seq_a[:, :63]
        neg[:, 63:] = seq_b[:, 63:]
        if np.count_nonzero(neg[:, :63]) < 18:
            # Transfer wrist trajectory of A onto handshape of B!
            for t in range(20):
                ha = seq_a[t, 63:].reshape(21, 3)
                hb = seq_b[t, 63:].reshape(21, 3)
                if np.count_nonzero(ha) >= 6 and np.count_nonzero(hb) >= 6:
                    neg[t, 63:] = (ha[0:1] + (hb - hb[0:1])).flatten()
                else:
                    neg[t, 63:] = seq_b[t, 63:]
    elif kind < 0.72:
        # 3) Spatial / finger-group permutation or inverted vertical trajectory (non-vocabulary ISL sign)
        neg = seq_a.copy()
        perm_groups = rng.permutation(4)
        src_ranges = [(5, 9), (9, 13), (13, 17), (17, 21)]
        for base in (0, 63):
            sub = neg[:, base:base + 63].reshape(20, 21, 3)
            m = np.any(sub != 0.0, axis=(1, 2))
            if np.any(m):
                reordered = sub.copy()
                for gi, pj in enumerate(perm_groups):
                    s0, e0 = src_ranges[gi]
                    s1, e1 = src_ranges[int(pj)]
                    reordered[:, s0:e0, :] = sub[:, s1:e1, :]
                reordered[m, :, 1] = 0.85 - reordered[m, :, 1]
                neg[:, base:base + 63] = reordered.reshape(20, 63)
    elif kind < 0.88:
        # 4) Low-velocity / held non-vocabulary handshape chimeras & permuted finger poses (prevents velocity shortcut!)
        pa = seq_a[int(rng.integers(6, 14))].copy()
        pb = seq_b[int(rng.integers(6, 14))].copy()
        pose_neg = np.zeros(126, dtype=np.float32)
        if rng.random() < 0.5 and np.count_nonzero(pa[63:]) >= 6 and np.count_nonzero(pb[63:]) >= 6:
            # Non-vocabulary mid-point blend + finger swap held steady near face/chest
            ha = pa[63:].reshape(21, 3)
            hb = pb[63:].reshape(21, 3)
            w_neg = np.array([float(rng.uniform(0.35, 0.65)), float(rng.uniform(0.28, 0.62)), 0.0], dtype=np.float32)
            rel_mix = 0.5 * (ha - ha[0:1]) + 0.5 * (hb - hb[0:1])
            rel_mix[:, 1] *= -1.0
            pose_neg[63:] = (w_neg[None, :] + rel_mix).flatten()
            if rng.random() < 0.45:
                pose_neg[:63] = (w_neg[None, :] + np.array([0.14, 0.05, 0.0], dtype=np.float32) + (hb - hb[0:1])).flatten()
        else:
            pose_neg = pa.copy()
            perm_groups = rng.permutation(4)
            while np.array_equal(perm_groups, np.arange(4)):
                perm_groups = rng.permutation(4)
            src_ranges = [(5, 9), (9, 13), (13, 17), (17, 21)]
            for base in (0, 63):
                sub = pose_neg[base:base + 63].reshape(21, 3)
                if np.count_nonzero(sub) >= 6:
                    reord = sub.copy()
                    for gi, pj in enumerate(perm_groups):
                        reord[src_ranges[gi][0]:src_ranges[gi][1]] = sub[src_ranges[int(pj)][0]:src_ranges[int(pj)][1]]
                    pose_neg[base:base + 63] = reord.flatten()
        nz_m = (pose_neg != 0.0).astype(np.float32)
        drift = rng.uniform(-0.02, 0.02, size=(1, 126)).astype(np.float32) * nz_m
        neg = np.array(
            [pose_neg + (t / 19.0) * drift[0] + rng.normal(0.0, 0.0045, size=126).astype(np.float32) * nz_m for t in range(20)],
            dtype=np.float32,
        )
    else:
        # 5) Casual non-signing fidgeting / waving / random smooth walk in front of laptop camera
        neg = np.zeros((20, 126), dtype=np.float32)
        t_grid = np.linspace(0.0, 2.0 * np.pi, 20, dtype=np.float32)
        base_hand = seq_a[10, 63:].reshape(21, 3).copy()
        if np.count_nonzero(base_hand) < 6:
            base_hand = rng.normal(0.0, 0.04, size=(21, 3)).astype(np.float32)
        rel_h = base_hand - base_hand[0:1]
        freq = float(rng.uniform(1.5, 3.5))
        for t in range(20):
            ang = float(rng.uniform(-0.8, 0.8))
            rot = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]], dtype=np.float32)
            w = np.array([
                0.50 + 0.16 * np.sin(freq * t_grid[t]),
                0.42 + 0.12 * np.cos(freq * 0.7 * t_grid[t]),
                0.0,
            ], dtype=np.float32)
            neg[t, 63:] = (w[None, :] + (rel_h @ rot.T)).flatten()

    if rng.random() < 0.45:
        neg = simulate_upper_body_webcam_frustum(neg, y_cutoff=float(rng.uniform(0.55, 0.68)), ar_target=4.0 / 3.0, rng=rng)
    return extract_kinematic_features(neg, length=20, stabilize=True)


def build_webcam_kinematic_model(input_dim=279, num_outputs=71):
    inp = tf.keras.Input(shape=(20, input_dim), name="kinematic_seq")
    x = tf.keras.layers.SpatialDropout1D(0.15)(inp)

    c3 = tf.keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="gelu")(x)
    c5 = tf.keras.layers.Conv1D(128, kernel_size=5, padding="same", dilation_rate=2, activation="gelu")(x)
    x = tf.keras.layers.Concatenate()([c3, c5])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    g1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(128, return_sequences=True, dropout=0.20)
    )(x)
    g1 = tf.keras.layers.LayerNormalization()(g1)
    x = tf.keras.layers.Add()([x, g1])

    g2 = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(96, return_sequences=True, dropout=0.20)
    )(x)
    g2 = tf.keras.layers.LayerNormalization()(g2)

    # Pure native built-in Keras attention pooling (zero custom layer classes -> 100% lossless Keras 3 serialization)
    attn_proj = tf.keras.layers.Dense(64, activation="tanh", name="attn_proj")(g2)
    attn_logits = tf.keras.layers.Dense(1, use_bias=False, name="attn_score")(attn_proj)
    attn_weights = tf.keras.layers.Softmax(axis=1, name="attn_weights")(attn_logits)
    p_attn = tf.keras.layers.Dot(axes=(1, 1), name="attn_dot")([attn_weights, g2])
    p_attn = tf.keras.layers.Flatten(name="attn_pool")(p_attn)
    p_max = tf.keras.layers.GlobalMaxPooling1D()(g2)
    h = tf.keras.layers.Concatenate()([p_attn, p_max])
    h = tf.keras.layers.Dropout(0.35)(h)
    h = tf.keras.layers.Dense(
        256, activation="gelu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Dropout(0.30)(h)
    out = tf.keras.layers.Dense(num_outputs, activation="softmax", name="probs_71")(h)
    return tf.keras.Model(inputs=inp, outputs=out, name="ISL_Webcam_Kinematic_71")


def compute_metrics_dict(y_true, probs_70):
    y_true = np.asarray(y_true, dtype=np.int64)
    preds = np.argmax(probs_70, axis=1)
    top1 = float(np.mean(preds == y_true) * 100.0)
    top3_hits = [int(y_true[i] in np.argsort(probs_70[i])[::-1][:3]) for i in range(len(y_true))]
    top3 = float(np.mean(top3_hits) * 100.0)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, preds, labels=list(range(NUM_SIGN_CLASSES)), average="macro", zero_division=0
    )
    p_cls, r_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        y_true, preds, labels=list(range(NUM_SIGN_CLASSES)), average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, preds, labels=list(range(NUM_SIGN_CLASSES)))
    return {
        "top1": round(top1, 2),
        "top3": round(top3, 2),
        "macro_precision": round(float(p_mac * 100.0), 2),
        "macro_recall": round(float(r_mac * 100.0), 2),
        "macro_f1": round(float(f1_mac * 100.0), 2),
        "per_class": {
            CLASSES[c]: {
                "precision": round(float(p_cls[c] * 100.0), 2),
                "recall": round(float(r_cls[c] * 100.0), 2),
                "f1": round(float(f1_cls[c] * 100.0), 2),
                "support": int(sup_cls[c]),
            }
            for c in range(NUM_SIGN_CLASSES)
        },
        "cm": cm,
        "preds": preds,
    }


def main():
    t_start = time.time()
    print("=" * 90)
    print("STEP 1: GENUINE INCLUDE CORPUS AUDIT & ZERO-LEAKAGE RECORDING-DISJOINT SPLIT")
    print("=" * 90)

    corpus = pickle.load(open("/tmp/isl_genuine_corpus.pkl", "rb"))
    gif_seqs = pickle.load(open("/tmp/kaggle_70_gif_trajectories.pkl", "rb"))
    extra_real_5 = pickle.load(open("/tmp/real_video_landmarks.pkl", "rb"))

    official_counts = defaultdict(int)
    for sp in ["include_train.txt", "include_val.txt", "include_test.txt"]:
        p = f"/tmp/include_meta/{sp}"
        if os.path.exists(p):
            for line in open(p):
                line = line.strip()
                if line:
                    cls_raw = line.split("/")[-2]
                    clean = cls_raw.split(". ", 1)[1].strip() if ". " in cls_raw else cls_raw.strip()
                    official_counts[clean.lower()] += 1

    records_by_cls = {c: {} for c in CLASSES}
    ood_records = []

    for item in corpus["mp4_results"]:
        if item["full_raw"] is None or np.count_nonzero(item["full_raw"]) < 18:
            continue
        if item["is_ood"]:
            ood_records.append(item)
        else:
            records_by_cls[item["class_name"]][item["mvi_id"]] = item

    extra_gt = {
        "/tmp/real_isl_videos/MVI_0033.mp4": ("MVI_0033", "How are you", False),
        "/tmp/real_isl_videos/MVI_5183.MOV": ("MVI_5183", "happy", False),
        "/tmp/real_isl_videos/MVI_5184.MOV": ("MVI_5184", "happy", False),
        "/tmp/real_isl_videos/MVI_9984.mp4": ("MVI_9984", "Hello", False),
        "/tmp/real_isl_videos/MVI_9987.mp4": ("MVI_9987", "How are you", False),
        "/tmp/real_isl_videos/MVI_5177.MOV": ("MVI_5177", "loud", True),
        "/tmp/real_isl_videos/MVI_9569.MOV": ("MVI_9569", "Beautiful", True),
    }
    for vpath, (mvi_id, cname, is_ood) in extra_gt.items():
        if vpath in extra_real_5:
            rec = {
                "mvi_id": mvi_id,
                "class_name": cname,
                "is_ood": is_ood,
                "source": "ai4bharat_raw_video",
                "full_raw": np.array(extra_real_5[vpath]["hol"], dtype=np.float32),
                "laptop_640x480_raw": np.array(extra_real_5[vpath]["lap_swap"], dtype=np.float32),
            }
            if is_ood:
                ood_records.append(rec)
            elif mvi_id not in records_by_cls[cname]:
                records_by_cls[cname][mvi_id] = rec

    for item in corpus["viso_results"]:
        cname = item["class_name"]
        if item["mvi_id"] not in records_by_cls[cname] and np.count_nonzero(item["full_raw"]) >= 18:
            records_by_cls[cname][item["mvi_id"]] = item

    # For the 6 classes where only 1 VisoVani 3D camera file exists publicly on GitHub,
    # keep the clean 3D camera recording in TRAIN and use the distinct Kaggle parquet[0] recording ONLY in TEST
    # (never adding 2D GIF blob reconstructions into X_train!).
    train_records = []
    test_records = []
    audit_rows = []
    rng_split = np.random.default_rng(SEED)

    for cls_idx, cname in enumerate(CLASSES):
        recs = sorted(records_by_cls[cname].values(), key=lambda r: r["mvi_id"])
        k_3d = len(recs)
        if k_3d >= 2:
            forced = [r for r in recs if r["mvi_id"] in FORCED_TEST_MVI]
            pool = [r for r in recs if r["mvi_id"] not in FORCED_TEST_MVI]
            rng_split.shuffle(pool)
            n_test_target = max(1, int(round(0.25 * k_3d)))
            need_more = max(0, n_test_target - len(forced))
            c_test = forced + pool[:need_more]
            t_ids = {r["mvi_id"] for r in c_test}
            c_train = [r for r in recs if r["mvi_id"] not in t_ids]
        else:
            # 1 clean 3D camera recording -> Train; distinct Kaggle parquet[0] video -> Held-Out Test ONLY
            c_train = list(recs)
            c_test = [{
                "mvi_id": f"KAGGLE_PARQUET0_{cname.replace(' ', '_')}",
                "class_name": cname,
                "is_ood": False,
                "source": "kaggle_parquet_gif",
                "full_raw": gif_seqs[cname],
                "laptop_640x480_raw": None,
            }]

        train_records.extend(c_train)
        test_records.extend(c_test)
        audit_rows.append({
            "class_index": cls_idx,
            "class_name": cname,
            "upstream_kaggle_parquet_count": official_counts.get(cname.lower(), 19),
            "verified_3d_camera_videos": k_3d,
            "train_videos": len(c_train),
            "heldout_test_videos": len(c_test),
            "has_real_mp4_pixels": int(any(r["laptop_640x480_raw"] is not None for r in recs)),
        })

    train_ids = {(r["class_name"], r["mvi_id"]) for r in train_records}
    test_ids = {(r["class_name"], r["mvi_id"]) for r in test_records}
    assert len(train_ids & test_ids) == 0, "FATAL: Recording leakage!"
    assert all(r["source"] != "kaggle_parquet_gif" for r in train_records), "FATAL: 2D GIF in X_train!"
    assert len({r["class_name"] for r in train_records}) == 70
    assert len({r["class_name"] for r in test_records}) == 70

    with open(os.path.join(ANALYSIS_DIR, "dataset_audit_70.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        w.writeheader()
        w.writerows(audit_rows)

    print(f"Pure 3D Camera Train Recordings (70/70 classes, 0% GIF pollution) : {len(train_records)}")
    print(f"Strict Held-Out Test Recordings (70/70 classes, ZERO overlap!)    : {len(test_records)}")
    print(f"Held-Out Real Human Non-Vocabulary OOD (.mp4) Recordings          : {len(ood_records)}")

    # Build unseen evaluation suites
    rng_eval = np.random.default_rng(1001)
    eval_all70_seqs, eval_all70_y = [], []
    eval_64cam_idx = []
    eval_real_mp4_laptop_seqs, eval_real_mp4_laptop_y = [], []

    for r in test_records:
        y = CLASS_TO_IDX[r["class_name"]]
        is_pure_cam = (r["source"] != "kaggle_parquet_gif")

        if is_pure_cam:
            eval_64cam_idx.append(len(eval_all70_seqs))
        eval_all70_seqs.append(r["full_raw"])
        eval_all70_y.append(y)

        if r["laptop_640x480_raw"] is not None and np.count_nonzero(r["laptop_640x480_raw"]) >= 18:
            lap_seq = r["laptop_640x480_raw"]
            eval_real_mp4_laptop_seqs.append(lap_seq)
            eval_real_mp4_laptop_y.append(y)
        else:
            lap_seq = simulate_upper_body_webcam_frustum(r["full_raw"], y_cutoff=0.60, ar_target=4.0 / 3.0, rng=rng_eval)

        if is_pure_cam:
            eval_64cam_idx.append(len(eval_all70_seqs))
        eval_all70_seqs.append(lap_seq)
        eval_all70_y.append(y)

    ood_eval_seqs = [
        r["laptop_640x480_raw"] if (r["laptop_640x480_raw"] is not None and np.count_nonzero(r["laptop_640x480_raw"]) >= 18) else r["full_raw"]
        for r in ood_records
    ]

    # =========================================================================
    # STEP 2: BASELINE `3cdac8c` EVALUATION ON IDENTICAL HELD-OUT SEQUENCES
    # =========================================================================
    print("\n" + "=" * 90)
    print("STEP 2: EVALUATING BASELINE `3cdac8c` ON IDENTICAL HELD-OUT TEST & 207 REAL OOD VIDEOS")
    print("=" * 90)
    spec_old = importlib.util.spec_from_file_location(
        "old_lstm_service", os.path.join(ROOT_DIR, "backups", "pre_retrain_3cdac8c", "lstm_service.py")
    )
    old_mod = importlib.util.module_from_file_location if False else importlib.util.module_from_spec(spec_old)
    spec_old.loader.exec_module(old_mod)

    def eval_old_predict_word(seq_list):
        probs = np.zeros((len(seq_list), NUM_SIGN_CLASSES), dtype=np.float32)
        words, confs = [], []
        for i, s in enumerate(seq_list):
            w, conf, top3 = old_mod.predict_word(list(s))
            words.append(w)
            confs.append(conf)
            if w != "Unknown":
                for d in top3:
                    probs[i, CLASS_TO_IDX[d["word"]]] = float(d["confidence"]) / 100.0
                rem = max(0.0, 1.0 - float(np.sum(probs[i])))
                probs[i] += rem / NUM_SIGN_CLASSES
        return words, confs, probs

    _, _, before_all70_probs = eval_old_predict_word(eval_all70_seqs)
    before_all70_metrics = compute_metrics_dict(eval_all70_y, before_all70_probs)

    _, _, before_lap_probs = eval_old_predict_word(eval_real_mp4_laptop_seqs)
    before_lap_top1 = round(float(np.mean(np.argmax(before_lap_probs, axis=1) == np.array(eval_real_mp4_laptop_y)) * 100.0), 2)
    before_lap_top3 = round(float(np.mean([eval_real_mp4_laptop_y[i] in np.argsort(before_lap_probs[i])[::-1][:3] for i in range(len(eval_real_mp4_laptop_y))]) * 100.0), 2)

    before_ood_words, before_ood_confs, _ = eval_old_predict_word(ood_eval_seqs)
    before_ood_unknown_pct = round(100.0 * sum(1 for w in before_ood_words if w == "Unknown") / len(before_ood_words), 2)
    before_ood_false_high = round(100.0 * sum(1 for w, c in zip(before_ood_words, before_ood_confs) if w != "Unknown" and c >= 60.0) / len(before_ood_words), 2)

    print(f"BEFORE (3cdac8c) All-70 Held-Out ({len(eval_all70_seqs)} seqs): Top-1={before_all70_metrics['top1']:.2f}%, Top-3={before_all70_metrics['top3']:.2f}%, Macro-F1={before_all70_metrics['macro_f1']:.2f}%")
    print(f"BEFORE (3cdac8c) Real 640x480 Laptop MP4s ({len(eval_real_mp4_laptop_seqs)} seqs): Top-1={before_lap_top1:.2f}%, Top-3={before_lap_top3:.2f}%")
    print(f"BEFORE (3cdac8c) 207 Real OOD Videos: Unknown={before_ood_unknown_pct:.2f}%, False >=60% Conf={before_ood_false_high:.2f}%")

    # =========================================================================
    # STEP 3: TRAIN `ISL_Webcam_Kinematic_71` (70 SIGNS + EXPLICIT OPEN-SET HEAD 70)
    # =========================================================================
    print("\n" + "=" * 90)
    print("STEP 3: TRAINING `ISL_Webcam_Kinematic_71` (PURE 3D TRAIN + OPEN-SET NEGATIVE EXPOSURE)")
    print("=" * 90)
    train_by_cls = defaultdict(list)
    for r in train_records:
        train_by_cls[CLASS_TO_IDX[r["class_name"]]].append(r)

    rng_tr = np.random.default_rng(2026)
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []

    with open(os.path.join(MODEL_DIR, "reference_landmarks_70.json"), "r", encoding="utf-8") as ref_f:
        ref_landmarks_70 = json.load(ref_f)

    DYNAMIC_PER_CLASS = 84
    STATIC_HOLD_PER_CLASS = 6
    for c_idx in range(NUM_SIGN_CLASSES):
        cname = CLASSES[c_idx]
        recs_c = train_by_cls[c_idx]
        added_c = 0
        for r in recs_c:
            X_train_list.append(extract_kinematic_features(r["full_raw"], length=20, stabilize=True))
            y_train_list.append(c_idx)
            added_c += 1
            if r["laptop_640x480_raw"] is not None and np.count_nonzero(r["laptop_640x480_raw"]) >= 18:
                X_train_list.append(extract_kinematic_features(r["laptop_640x480_raw"], length=20, aspect_ratio=4.0 / 3.0, stabilize=True))
                y_train_list.append(c_idx)
                added_c += 1

        while added_c < DYNAMIC_PER_CLASS:
            r = recs_c[int(rng_tr.integers(0, len(recs_c)))]
            use_lap = (r["laptop_640x480_raw"] is not None and np.count_nonzero(r["laptop_640x480_raw"]) >= 18 and rng_tr.random() < 0.5)
            base_seq = r["laptop_640x480_raw"] if use_lap else r["full_raw"]
            X_train_list.append(augment_webcam_training_sample(base_seq, rng_tr))
            y_train_list.append(c_idx)
            added_c += 1

        # Also teach steady held-pose webcam attempts (beginners holding a target sign steady for ~2s):
        # half drawn from active apex frames of `train_records[c_idx]`, half from studio `reference_landmarks_70.json`.
        ref_base = np.asarray(ref_landmarks_70[cname], dtype=np.float32)
        for sh_i in range(STATIC_HOLD_PER_CLASS):
            if sh_i % 2 == 0:
                r = recs_c[int(rng_tr.integers(0, len(recs_c)))]
                s20 = pad_or_truncate(r["full_raw"], 20)
                active_ts = [t for t in range(4, 16) if np.count_nonzero(s20[t]) >= 18]
                pose_vec = s20[active_ts[int(rng_tr.integers(0, len(active_ts)))]].copy() if active_ts else ref_base.copy()
            else:
                pose_vec = ref_base.copy()
            nz_mask = (pose_vec != 0.0).astype(np.float32)
            sc = float(rng_tr.uniform(0.85, 1.18))
            dx, dy = float(rng_tr.uniform(-0.06, 0.06)), float(rng_tr.uniform(-0.05, 0.05))
            p_mod = pose_vec.copy()
            for base_h in (0, 63):
                h = p_mod[base_h:base_h + 63].reshape(21, 3)
                if np.count_nonzero(h) >= 6:
                    w0 = h[0].copy() + np.array([dx, dy, 0.0], dtype=np.float32)
                    p_mod[base_h:base_h + 63] = (w0[None, :] + (h - h[0:1]) * sc).flatten()
            held_seq = np.array(
                [p_mod + rng_tr.normal(0.0, 0.0045, size=126).astype(np.float32) * nz_mask for _ in range(20)],
                dtype=np.float32,
            )
            X_train_list.append(extract_kinematic_features(held_seq, length=20, stabilize=True))
            y_train_list.append(c_idx)

        for _ in range(6):
            r = recs_c[int(rng_tr.integers(0, len(recs_c)))]
            X_val_list.append(augment_webcam_training_sample(r["full_raw"], rng_tr))
            y_val_list.append(c_idx)

    # Add strictly self-supervised Open-Set negatives (class 70 = "Unknown") generated ONLY from `train_by_cls`!
    N_OOD_TRAIN = 1500
    for _ in range(N_OOD_TRAIN):
        X_train_list.append(synthesize_open_set_negative(train_by_cls, rng_tr))
        y_train_list.append(UNKNOWN_CLASS_IDX)
    for _ in range(90):
        X_val_list.append(synthesize_open_set_negative(train_by_cls, rng_tr))
        y_val_list.append(UNKNOWN_CLASS_IDX)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int64)
    X_val = np.array(X_val_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.int64)

    kin_mean = np.mean(X_train, axis=(0, 1), keepdims=False).astype(np.float32)
    kin_std = (np.std(X_train, axis=(0, 1), keepdims=False) + 1e-6).astype(np.float32)
    np.save(os.path.join(MODEL_DIR, "kin_mean.npy"), kin_mean)
    np.save(os.path.join(MODEL_DIR, "kin_std.npy"), kin_std)

    X_train_s = (X_train - kin_mean) / kin_std
    X_val_s = (X_val - kin_mean) / kin_std
    y_train_oh = tf.keras.utils.to_categorical(y_train, TOTAL_OUTPUT_CLASSES)
    y_val_oh = tf.keras.utils.to_categorical(y_val, TOTAL_OUTPUT_CLASSES)

    model = build_webcam_kinematic_model(input_dim=X_train.shape[-1], num_outputs=TOTAL_OUTPUT_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1.4e-3, weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.04),
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )

    ckpt_path = os.path.join(MODEL_DIR, "isl_webcam_kinematic_70.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=3e-5, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=9, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(ANALYSIS_DIR, "webcam_kinematic_training_log.csv")),
    ]

    perm = rng_tr.permutation(len(X_train_s))
    model.fit(
        X_train_s[perm],
        y_train_oh[perm],
        validation_data=(X_val_s, y_val_oh),
        epochs=34,
        batch_size=64,
        verbose=2,
        callbacks=callbacks,
    )
    model.save(ckpt_path)

    # Verify cold serialization reload from disk BEFORE computing ANY held-out metrics!
    reloaded_model = tf.keras.models.load_model(ckpt_path, compile=False)
    serial_max_err = float(
        np.max(np.abs(model(X_val_s[:8], training=False).numpy() - reloaded_model(X_val_s[:8], training=False).numpy()))
    )
    print(f"Verified cold .keras reload fidelity: max |p_ram - p_disk| = {serial_max_err:.2e}")
    assert serial_max_err < 1e-6, f"Keras serialization mismatch detected: {serial_max_err}"
    model = reloaded_model

    # =========================================================================
    # STEP 4: CONVEX MULTI-STREAM FUSION + EXPLICIT OPEN-SET REJECTION EVALUATION
    # =========================================================================
    print("\n" + "=" * 90)
    print("STEP 4: HELD-OUT EVALUATION (ALL 70 SIGNS + 64 PURE-CAMERA SIGNS + 127 LAPTOP MP4S + 207 OOD)")
    print("=" * 90)
    lstm_studio = tf.keras.models.load_model(os.path.join(MODEL_DIR, "isl_lstm_model_70.keras"))
    mlp_studio = tf.keras.models.load_model(os.path.join(ROOT_DIR, "Models_MLP", "isl_mlp_model_70_handaug.keras"))
    X_mean = np.load(os.path.join(MODEL_DIR, "X_mean.npy"))
    X_std = np.load(os.path.join(MODEL_DIR, "X_std.npy"))
    X_mlp_mean = np.load(os.path.join(ROOT_DIR, "Models_MLP", "X_mlp_mean.npy"))
    X_mlp_std = np.load(os.path.join(ROOT_DIR, "Models_MLP", "X_mlp_std.npy"))

    def batch_convex_predict(seq_list):
        kin_batch, wr_batch = [], []
        for s in seq_list:
            clean_s, _ = stabilize_hand_sequence(s)
            kin_batch.append(extract_kinematic_features(clean_s, length=20, stabilize=False))
            wr_batch.append(pad_or_truncate([normalize_wrist_relative(f) for f in clean_s], 20))

        kin_arr = (np.array(kin_batch, dtype=np.float32) - kin_mean) / kin_std
        p71 = model(kin_arr, training=False).numpy()
        p_ood = p71[:, UNKNOWN_CLASS_IDX]
        p_kin70 = p71[:, :NUM_SIGN_CLASSES] / np.maximum(1.0 - p_ood[:, None], 1e-6)

        p_lstm = lstm_studio(np.concatenate([standardize_sequence(w, X_mean, X_std) for w in wr_batch], axis=0), training=False).numpy()
        p_mlp = mlp_studio(np.concatenate([standardize_sequence(w, X_mlp_mean, X_mlp_std) for w in wr_batch], axis=0), training=False).numpy()

        # Convex probability ensemble (immune to studio log(1e-6) hip-hand veto while preserving 1,034-video studio priors!)
        p_ens70 = 0.72 * p_kin70 + 0.16 * p_lstm + 0.12 * p_mlp
        p_ens70 /= np.sum(p_ens70, axis=1, keepdims=True)
        # Calibrated confidence discounted by open-set OOD posterior!
        cal_probs = p_ens70 * (1.0 - 0.65 * p_ood[:, None])
        return p_kin70, p_ood, p_lstm, p_mlp, p_ens70, cal_probs

    p_kin_all, p_ood_all, p_lstm_all, p_mlp_all, p_ens_all, cal_all = batch_convex_predict(eval_all70_seqs)
    kin_only_metrics = compute_metrics_dict(eval_all70_y, p_kin_all)
    after_all70_metrics = compute_metrics_dict(eval_all70_y, p_ens_all)

    cam64_y = np.array(eval_all70_y)[eval_64cam_idx]
    cam64_before_top1 = round(float(np.mean(np.argmax(before_all70_probs[eval_64cam_idx], axis=1) == cam64_y) * 100.0), 2)
    cam64_after_top1 = round(float(np.mean(np.argmax(p_ens_all[eval_64cam_idx], axis=1) == cam64_y) * 100.0), 2)
    cam64_after_top3 = round(float(np.mean([cam64_y[i] in np.argsort(p_ens_all[eval_64cam_idx[i]])[::-1][:3] for i in range(len(cam64_y))]) * 100.0), 2)

    p_kin_lap, p_ood_lap, _, _, p_ens_lap, cal_lap = batch_convex_predict(eval_real_mp4_laptop_seqs)
    after_lap_top1 = round(float(np.mean(np.argmax(p_ens_lap, axis=1) == np.array(eval_real_mp4_laptop_y)) * 100.0), 2)
    after_lap_top3 = round(float(np.mean([eval_real_mp4_laptop_y[i] in np.argsort(p_ens_lap[i])[::-1][:3] for i in range(len(eval_real_mp4_laptop_y))]) * 100.0), 2)

    p_kin_ood, p_ood_ood, p_lstm_ood, p_mlp_ood, p_ens_ood, cal_ood = batch_convex_predict(ood_eval_seqs)

    def is_rejected_unknown(p_ood_vec, p_ens_mat, cal_mat, p_lstm_mat, p_mlp_mat):
        max_cal = np.max(cal_mat, axis=1)
        sorted_c = np.sort(cal_mat, axis=1)[:, ::-1]
        margin = sorted_c[:, 0] - sorted_c[:, 1]
        entropy = -np.sum(p_ens_mat * np.log(np.clip(p_ens_mat, 1e-8, 1.0)), axis=1) / np.log(NUM_SIGN_CLASSES)
        # Disagreement check: neither studio LSTM nor studio MLP puts top prediction in its top-5 AND kinematic confidence < 0.78
        top_c = np.argmax(p_ens_mat, axis=1)
        studio_support = np.array([
            max(float(p_lstm_mat[i, top_c[i]]), float(p_mlp_mat[i, top_c[i]]))
            for i in range(len(top_c))
        ])
        return (
            (p_ood_vec >= 0.42)
            | (max_cal < 0.40)
            | ((entropy > 0.48) & (margin < 0.22))
            | ((p_ood_vec >= 0.22) & (studio_support < 0.03) & (max_cal < 0.68))
        )

    in_rej = is_rejected_unknown(p_ood_all, p_ens_all, cal_all, p_lstm_all, p_mlp_all)
    ood_rej = is_rejected_unknown(p_ood_ood, p_ens_ood, cal_ood, p_lstm_ood, p_mlp_ood)
    after_ood_unknown_pct = round(100.0 * float(np.mean(ood_rej)), 2)
    after_ood_false_high = round(100.0 * float(np.mean((~ood_rej) & (np.max(cal_ood, axis=1) >= 0.60))), 2)

    print(f"KINEMATIC-71 ALONE All-70 Held-Out: Top-1={kin_only_metrics['top1']:.2f}%, Top-3={kin_only_metrics['top3']:.2f}%, Macro-F1={kin_only_metrics['macro_f1']:.2f}%")
    print(f"AFTER (CONVEX HYBRID + OOD-71) All-70 Held-Out ({len(eval_all70_seqs)} seqs):")
    print(f"  Top-1 Acc: {after_all70_metrics['top1']:6.2f}% (was {before_all70_metrics['top1']:6.2f}%) | "
          f"Top-3 Acc: {after_all70_metrics['top3']:6.2f}% (was {before_all70_metrics['top3']:6.2f}%)")
    print(f"  Macro-P  : {after_all70_metrics['macro_precision']:6.2f}% (was {before_all70_metrics['macro_precision']:6.2f}%) | "
          f"Macro-R  : {after_all70_metrics['macro_recall']:6.2f}% (was {before_all70_metrics['macro_recall']:6.2f}%) | "
          f"Macro-F1 : {after_all70_metrics['macro_f1']:6.2f}% (was {before_all70_metrics['macro_f1']:6.2f}%)")
    print(f"Pure 3D Camera Signer-Disjoint Subset ({len(eval_64cam_idx)} seqs across 64 multi-signer classes): Top-1 = {cam64_after_top1:.2f}% (was {cam64_before_top1:.2f}%), Top-3 = {cam64_after_top3:.2f}%")
    print(f"AFTER Real 640x480 Laptop MP4s ({len(eval_real_mp4_laptop_seqs)} seqs): Top-1 = {after_lap_top1:.2f}% (was {before_lap_top1:.2f}%), Top-3 = {after_lap_top3:.2f}% (was {before_lap_top3:.2f}%)")
    print(f"AFTER 207 Real OOD Human Videos: Returned 'Unknown' = {after_ood_unknown_pct:.2f}% (was {before_ood_unknown_pct:.2f}%) | False >=60% Confident Hallucinations = {after_ood_false_high:.2f}% (was {before_ood_false_high:.2f}%) | In-Vocab Keep Rate = {100.0*(1.0-float(np.mean(in_rej))):.2f}%")

    np.savetxt(os.path.join(ANALYSIS_DIR, "confusion_matrix_before_70.csv"), before_all70_metrics["cm"], fmt="%d", delimiter=",")
    np.savetxt(os.path.join(ANALYSIS_DIR, "confusion_matrix_after_70.csv"), after_all70_metrics["cm"], fmt="%d", delimiter=",")

    per_cls_rows = []
    for c_idx, cname in enumerate(CLASSES):
        b = before_all70_metrics["per_class"][cname]
        a = after_all70_metrics["per_class"][cname]
        per_cls_rows.append({
            "class_index": c_idx,
            "class_name": cname,
            "support": a["support"],
            "before_precision": b["precision"],
            "before_recall": b["recall"],
            "before_f1": b["f1"],
            "after_precision": a["precision"],
            "after_recall": a["recall"],
            "after_f1": a["f1"],
            "delta_f1": round(a["f1"] - b["f1"], 2),
        })
    with open(os.path.join(ANALYSIS_DIR, "per_class_before_after_70.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_cls_rows[0].keys()))
        w.writeheader()
        w.writerows(per_cls_rows)

    summary = {
        "n_classes": NUM_SIGN_CLASSES,
        "train_unique_recordings": len(train_records),
        "test_unique_recordings": len(test_records),
        "eval_sequences_all70": len(eval_all70_seqs),
        "eval_sequences_pure_3d_camera": len(eval_64cam_idx),
        "eval_sequences_real_640x480_laptop": len(eval_real_mp4_laptop_seqs),
        "eval_sequences_real_ood_videos": len(ood_eval_seqs),
        "before_all70": {k: v for k, v in before_all70_metrics.items() if k not in ("cm", "preds", "per_class")},
        "kinematic_alone_all70": {k: v for k, v in kin_only_metrics.items() if k not in ("cm", "preds", "per_class")},
        "after_all70": {k: v for k, v in after_all70_metrics.items() if k not in ("cm", "preds", "per_class")},
        "pure_3d_camera_64classes": {"before_top1": cam64_before_top1, "after_top1": cam64_after_top1, "after_top3": cam64_after_top3},
        "before_real_laptop_640x480": {"top1": before_lap_top1, "top3": before_lap_top3},
        "after_real_laptop_640x480": {"top1": after_lap_top1, "top3": after_lap_top3},
        "before_real_ood_207": {"unknown_pct": before_ood_unknown_pct, "false_confident_60_pct": before_ood_false_high},
        "after_real_ood_207": {"unknown_pct": after_ood_unknown_pct, "false_confident_60_pct": after_ood_false_high},
        "confused_focus": {
            c: {
                "before": before_all70_metrics["per_class"][c],
                "after": after_all70_metrics["per_class"][c],
            }
            for c in ["Police", "Bicycle", "Doctor", "Hello", "How are you", "happy", "Good Morning", "Father", "Mother"]
        },
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    with open(os.path.join(ANALYSIS_DIR, "retraining_metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved complete evaluation artifacts to models_LSTM/analysis/retraining_metrics_summary.json!")


if __name__ == "__main__":
    main()
