import numpy as np

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# Gesture instruction hints for all 70 classes (from Cell 25 of isl-sign-language.ipynb)
GESTURE_HINTS = {
    'Hello': 'Open palm facing camera | All 5 fingers spread wide | Single hand',
    'Thank you': 'Flat hand starts at chin | Move hand forward | Single hand',
    'How are you': 'Both hands open | Palms facing up | Move alternately',
    'Good Morning': 'Flat hand at waist | Rise upward slowly | Single hand',
    'Good night': 'Open hand | Close fingers downward | Like sleeping gesture',
    'happy': 'Flat hand on chest | Circular motion upward | Single hand',
    'good': 'Thumbs up clearly | Thumb pointing straight up | Single hand',
    'bad': 'Thumbs down clearly | Thumb pointing straight down | Single hand',
    'Pleased': 'Both hands flat | Press gently to chest | Both hands',
    'Alright': 'Thumb and index form circle | Other fingers straight up | Single hand',
    'cold': 'Both fists closed | Shake side to side | Both hands',
    'Mother': 'Thumb touches chin | Other 4 fingers spread open | Single hand',
    'Father': 'Thumb touches forehead | Other 4 fingers spread open | Single hand',
    'Brother': 'Index fingers side by side | Hold horizontally | Both hands',
    'Sister': 'Index fingers side by side | Move both downward | Both hands',
    'Son': 'Flat hand at waist level | Palm facing down | Single hand',
    'Daughter': 'Flat hand at chest level | Palm facing down | Single hand',
    'Grandfather': 'Closed fist near forehead | Move forward twice | Single hand',
    'Grandmother': 'Closed fist near chin | Move forward twice | Single hand',
    'Husband': 'Both hands interlocked | Hold at chest level | Both hands',
    'Wife': 'Both hands interlocked | Gentle shake at chest | Both hands',
    'Doctor': 'Two fingers on wrist | Tap twice like checking pulse | Single hand',
    'Teacher': 'Flat hands open near temples | Open and close twice | Both hands',
    'Police': 'Fist near shoulder | Thumb up like badge | Single hand',
    'Soldier': 'Flat hand to forehead | Clean salute gesture | Single hand',
    'Lawyer': 'Both hands together | Like holding a book | Both hands',
    'Reporter': 'Closed fist near mouth | Like holding microphone | Single hand',
    'Manager': 'Index finger point up | Move in small circle | Single hand',
    'Waiter': 'Flat palm facing up | Slight forward bow | Single hand',
    'School': 'Both hands flat | Clap twice clearly | Both hands',
    'Hospital': 'Cross shape with index fingers | Hold clearly | Both hands',
    'Market': 'Rub thumb over fingers | Like counting money | Single hand',
    'Restaurant': 'Hand to mouth | Eating gesture twice | Single hand',
    'Library': 'Both hands flat | Hold like open book | Both hands',
    'Temple': 'Both hands together | Prayer position at chest | Both hands',
    'University': 'Both hands open | Sweep wide apart | Both hands',
    'Bank': 'Rub thumb over fingers | Money sign clearly | Single hand',
    'Dog': 'Pat thigh | Then snap fingers | Single hand',
    'Cat': 'Pinch fingers near cheek | Like cat whiskers | Single hand',
    'Horse': 'Two fingers up like ears | Bounce hand twice | Single hand',
    'Bird': 'Index and thumb together | Open close like beak | Single hand',
    'Fish': 'Flat hand wave | Side to side like fish | Single hand',
    'Cow': 'Pinky and index up | Like horns near head | Single hand',
    'Red': 'Index finger | Brush down on lips twice | Single hand',
    'Blue': 'B handshape | Index and thumb shake | Single hand',
    'Green': 'G handshape | Shake near chest | Single hand',
    'Yellow': 'Y handshape | Pinky and thumb out shake | Single hand',
    'Black': 'Index finger | Swipe across forehead | Single hand',
    'White': 'Open hand on chest | Pull out to closed fist | Single hand',
    'Book': 'Both flat hands together | Open like a book | Both hands',
    'Pen': 'Pinch fingers like holding pen | Write in air | Single hand',
    'Table': 'Both forearms flat | Tap one on other twice | Both hands',
    'Chair': 'Two fingers | Sit on other two fingers | Both hands',
    'Door': 'B handshape | Rotate wrist like opening door | Single hand',
    'Window': 'Both flat hands | Slide one up and down | Both hands',
    'Telephone': 'Y handshape near ear | Like holding phone | Single hand',
    'Computer': 'C handshape | Move forward near forehead | Single hand',
    'Car': 'Both fists | Steer like driving a wheel | Both hands',
    'Bus': 'Both hands grip | Like holding bus handle bar | Both hands',
    'Train': 'Two fingers on two fingers | Rub back and forth | Both hands',
    'Bicycle': 'Both fists | Rotate alternately like pedaling | Both hands',
    'Plane': 'Y handshape with index up | Move forward | Single hand',
    'Morning': 'Flat hand at elbow | Rise upward like sun | Single hand',
    'Afternoon': 'Flat forearm | Angle like afternoon sun position | Single hand',
    'Evening': 'Flat hand | Angle downward like setting sun | Single hand',
    'Today': 'Both Y handshapes | Drop down simultaneously | Both hands',
    'Tomorrow': 'Thumb from chin | Move forward once | Single hand',
    'I': 'Point index finger | Clearly point to yourself | Single hand',
    'you': 'Point index finger | Clearly point forward to other person | Single hand',
    'we': 'Index finger | Sweep from one shoulder to other | Single hand',
}

CANONICAL_ASPECT_RATIO = 16.0 / 9.0


def _mirror_single_hand_x(hand_21x3, is_raw):
    flipped = hand_21x3.copy()
    if is_raw:
        flipped[:, 0] = 1.0 - flipped[:, 0]
    else:
        flipped[:, 0] = -flipped[:, 0]
    return flipped


def stabilize_hand_sequence(sequence, aspect_ratio=None):
    """
    Canonicalizes raw or wrist-centered MediaPipe landmark sequences so live webcam
    streams match full-body 16:9 Holistic training conventions:
      1. Corrects non-16:9 camera aspect ratios around frame center x=0.5 before L_inf scaling.
      2. Suppresses duplicate MediaPipe detections firing on the same physical wrist (< 0.065).
      3. Orders true 2-hand raw detections spatially in unmirrored camera space
         (smaller x = signer's right_hand [63:126]; larger x = signer's left_hand [0:63]).
      4. Prevents single-hand signs from teleporting between slots across frames on transient
         handedness flips, and maps pure left-hand-only 1-hand signs into canonical right_hand
         via horizontal chirality reflection (avoiding the 2.86% idle-left-slot collapse).
    Returns:
      clean_seq: np.ndarray of shape (T, 126), dtype float32 (in raw space if input is raw).
      active_indices: list of frame indices containing at least one valid hand.
    """
    if sequence is None or len(sequence) == 0:
        return np.zeros((0, 126), dtype=np.float32), []

    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 126:
        raise ValueError(f"Expected 126 landmark features per frame, got {arr.shape[1]}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).copy()
    n_frames = arr.shape[0]

    # Detect whether coordinates are in raw image space [0,1] vs already wrist-centered (wrist == 0)
    wrist_norms = np.linalg.norm(arr[:, [0, 1]], axis=1) + np.linalg.norm(arr[:, [63, 64]], axis=1)
    is_raw = bool(np.any(wrist_norms > 1e-3))

    if is_raw and aspect_ratio is not None and aspect_ratio > 0:
        ar_scale = float(aspect_ratio) / CANONICAL_ASPECT_RATIO
        if abs(ar_scale - 1.0) > 1e-3:
            for base in (0, 63):
                sub = arr[:, base:base + 63].reshape(n_frames, 21, 3)
                mask = np.any(sub != 0.0, axis=(1, 2))
                if np.any(mask):
                    sub[mask, :, 0] = 0.5 + (sub[mask, :, 0] - 0.5) * ar_scale
                    sub[mask, :, 2] *= ar_scale
                arr[:, base:base + 63] = sub.reshape(n_frames, 63)

    per_frame = []
    active_indices = []
    two_hand_count = 0

    for t in range(n_frames):
        lh = arr[t, :63].reshape(21, 3)
        rh = arr[t, 63:].reshape(21, 3)
        has_l = int(np.count_nonzero(lh)) >= 6
        has_r = int(np.count_nonzero(rh)) >= 6

        if has_l and has_r and is_raw:
            w_dist = float(np.linalg.norm(lh[0, :2] - rh[0, :2]))
            # Also check centroid overlap in case wrists overlap slightly during a cross gesture
            c_dist = float(np.linalg.norm(np.mean(lh[:, :2], axis=0) - np.mean(rh[:, :2], axis=0)))
            if w_dist < 0.055 and c_dist < 0.055:
                # Duplicate detection on the exact same physical hand; keep dominant slot
                has_l = False

        if has_l or has_r:
            active_indices.append(t)
        if has_l and has_r:
            two_hand_count += 1
        per_frame.append((has_l, lh, has_r, rh))

    if not active_indices:
        return np.zeros((n_frames, 126), dtype=np.float32), []

    is_two_handed = (two_hand_count >= 2) or (two_hand_count / len(active_indices) >= 0.20)
    out = np.zeros((n_frames, 126), dtype=np.float32)

    if not is_two_handed:
        # Pure single-hand sign across the sequence: check which physical hand dominated
        l_votes = sum(1 for (hl, _, hr, _) in per_frame if hl and not hr)
        r_votes = sum(1 for (hl, _, hr, _) in per_frame if hr and not hl)
        dominant_is_left = l_votes > r_votes

        for t, (hl, lh, hr, rh) in enumerate(per_frame):
            if hr:
                out[t, 63:] = rh.flatten()
            elif hl:
                # If the user signed entirely with their left hand, mirror into canonical right_hand;
                # if it was a 1-2 frame MediaPipe label flip during a right-hand sign, DO NOT mirror!
                h_use = _mirror_single_hand_x(lh, is_raw) if dominant_is_left else lh
                out[t, 63:] = h_use.flatten()
    else:
        last_l_wrist = None
        last_r_wrist = None
        for t, (hl, lh, hr, rh) in enumerate(per_frame):
            if hl and hr:
                if is_raw:
                    # Unmirrored camera view: smaller x is signer's right hand [63:126], larger x is signer's left hand [0:63]
                    if lh[0, 0] < rh[0, 0] - 0.04:
                        rh_use, lh_use = lh, rh
                    else:
                        lh_use, rh_use = lh, rh
                else:
                    lh_use, rh_use = lh, rh
                out[t, :63] = lh_use.flatten()
                out[t, 63:] = rh_use.flatten()
                last_l_wrist = lh_use[0, :2].copy()
                last_r_wrist = rh_use[0, :2].copy()
            elif hr or hl:
                h_only = rh if hr else lh
                if is_raw and last_l_wrist is not None and last_r_wrist is not None:
                    dl = float(np.linalg.norm(h_only[0, :2] - last_l_wrist))
                    dr = float(np.linalg.norm(h_only[0, :2] - last_r_wrist))
                    if dl < dr:
                        out[t, :63] = h_only.flatten()
                        last_l_wrist = h_only[0, :2].copy()
                    else:
                        out[t, 63:] = h_only.flatten()
                        last_r_wrist = h_only[0, :2].copy()
                else:
                    if hr:
                        out[t, 63:] = h_only.flatten()
                    else:
                        out[t, :63] = h_only.flatten()

    return out, active_indices


def _hand_palm_scale(hand_21x3):
    """Scale reference in image space for camera-distance invariant kinematics."""
    d9 = float(np.linalg.norm(hand_21x3[9, :2] - hand_21x3[0, :2]))
    d5_17 = float(np.linalg.norm(hand_21x3[5, :2] - hand_21x3[17, :2]))
    return max(d9, d5_17, 0.04)


def extract_kinematic_features(sequence, length=20, aspect_ratio=None, stabilize=True):
    """
    Extracts a 279-dimensional scale-, position-, and aspect-invariant spatio-temporal
    feature tensor of shape (`length`, 279) from a raw or wrist-centered landmark sequence:
      [0:126]   : Stabilized wrist-relative 3D hand coordinates (L_inf normalized per hand)
      [126:252] : Frame-to-frame finger articulation velocity (wr[t] - wr[t-1])
      [252:258] : Camera-distance-invariant 3D wrist velocity (delta_wrist / median_palm_size)
      [258:263] : Inter-hand relative 3D vector ((wrist_L - wrist_R) / median_palm_size) + [has_L, has_R]
      [263:269] : 3D unit palm normal vectors for left and right hands
      [269:279] : Anatomical pinch & per-finger curl distances for left (5) and right (5) hands
    """
    if sequence is None or len(sequence) == 0:
        return np.zeros((length, 279), dtype=np.float32)

    if stabilize:
        clean_raw, active_idx = stabilize_hand_sequence(sequence, aspect_ratio=aspect_ratio)
    else:
        clean_raw = np.asarray(sequence, dtype=np.float32)
        if clean_raw.ndim == 1:
            clean_raw = clean_raw.reshape(1, 126)
        active_idx = [i for i in range(len(clean_raw)) if np.count_nonzero(clean_raw[i]) >= 6]

    if not active_idx:
        return np.zeros((length, 279), dtype=np.float32)

    # Trim dead camera lead-in / lead-out padding softly (keep 1 boundary frame when present)
    first_i = max(0, active_idx[0] - 1)
    last_i = min(len(clean_raw) - 1, active_idx[-1] + 1)
    trimmed = clean_raw[first_i:last_i + 1]
    resampled_raw = pad_or_truncate(trimmed, length=length)

    is_raw = bool(
        np.any(np.linalg.norm(resampled_raw[:, [0, 1]], axis=1) > 1e-3)
        or np.any(np.linalg.norm(resampled_raw[:, [63, 64]], axis=1) > 1e-3)
    )

    palm_scales = []
    if is_raw:
        for t in range(length):
            for base in (0, 63):
                h = resampled_raw[t, base:base + 63].reshape(21, 3)
                if np.count_nonzero(h) >= 6:
                    palm_scales.append(_hand_palm_scale(h))
    median_palm = float(np.median(palm_scales)) if palm_scales else 0.08

    wr = np.array([normalize_wrist_relative(resampled_raw[t]) for t in range(length)], dtype=np.float32)
    wr_vel = np.zeros_like(wr)
    wr_vel[1:] = wr[1:] - wr[:-1]

    wrist_vel = np.zeros((length, 6), dtype=np.float32)
    inter_hand = np.zeros((length, 5), dtype=np.float32)
    palm_normals = np.zeros((length, 6), dtype=np.float32)
    curls = np.zeros((length, 10), dtype=np.float32)

    prev_wl = None
    prev_wr = None

    for t in range(length):
        lh_raw = resampled_raw[t, :63].reshape(21, 3)
        rh_raw = resampled_raw[t, 63:].reshape(21, 3)
        hl = np.count_nonzero(lh_raw) >= 6
        hr = np.count_nonzero(rh_raw) >= 6
        inter_hand[t, 3] = 1.0 if hl else 0.0
        inter_hand[t, 4] = 1.0 if hr else 0.0

        if is_raw:
            if hl:
                if prev_wl is not None:
                    wrist_vel[t, 0:3] = np.clip((lh_raw[0] - prev_wl) / median_palm, -3.0, 3.0)
                prev_wl = lh_raw[0].copy()
            else:
                prev_wl = None

            if hr:
                if prev_wr is not None:
                    wrist_vel[t, 3:6] = np.clip((rh_raw[0] - prev_wr) / median_palm, -3.0, 3.0)
                prev_wr = rh_raw[0].copy()
            else:
                prev_wr = None

            if hl and hr:
                inter_hand[t, 0:3] = np.clip((lh_raw[0] - rh_raw[0]) / median_palm, -5.0, 5.0)

        for h_idx, base in enumerate((0, 63)):
            h_wr = wr[t, base:base + 63].reshape(21, 3)
            if np.count_nonzero(h_wr) >= 6:
                v1 = h_wr[5] - h_wr[0]
                v2 = h_wr[17] - h_wr[0]
                nrm = np.cross(v1, v2)
                n_len = float(np.linalg.norm(nrm))
                if n_len > 1e-6:
                    palm_normals[t, h_idx * 3:(h_idx + 1) * 3] = nrm / n_len
                c_off = h_idx * 5
                curls[t, c_off + 0] = float(np.linalg.norm(h_wr[4] - h_wr[8]))
                curls[t, c_off + 1] = float(np.linalg.norm(h_wr[8] - h_wr[0]))
                curls[t, c_off + 2] = float(np.linalg.norm(h_wr[12] - h_wr[0]))
                curls[t, c_off + 3] = float(np.linalg.norm(h_wr[16] - h_wr[0]))
                curls[t, c_off + 4] = float(np.linalg.norm(h_wr[20] - h_wr[0]))

    feats = np.concatenate([wr, wr_vel, wrist_vel, inter_hand, palm_normals, curls], axis=1)
    return np.nan_to_num(feats.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def normalize_wrist_relative(landmarks_126):
    """
    Exact wrist-relative normalization from notebooks (Cell 5 MLP / Cell 9 LSTM).
    For each hand (Left 0:63, Right 63:126):
      wrist = hand[0]
      hand = (hand - wrist) / (max(abs(hand - wrist)) + 1e-8)
    Absent hand remains 0.0.
    """
    lm = np.array(landmarks_126, dtype=np.float32).copy()
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

def pad_or_truncate(sequence, length=20):
    """
    Temporal resampling to exactly `length` frames using linspace.
    Always uses linspace-based resampling regardless of sequence length —
    this eliminates zero-padding entirely. Zero-padding caused the LSTM to
    always predict a biased class ('Horse') when detection was sparse.
    Works correctly for any input length >= 1.
    """
    if len(sequence) == 0:
        return np.zeros((length, 126), dtype=np.float32)
    seq = np.array(sequence, dtype=np.float32)
    # Always linspace: spreads frames evenly across target length (no zeros ever)
    indices = np.linspace(0, len(seq) - 1, length, dtype=int)
    return seq[indices]

def standardize_sequence(sequence, mean, std):
    """
    Global Z-score standardization: (X - mean) / (std + 1e-8).
    """
    seq = np.array(sequence, dtype=np.float32)
    norm = (seq - mean) / (std + 1e-8)
    return np.nan_to_num(norm, nan=0.0)

def hand_dist(a, b):
    """Mean Euclidean distance between 21 3D points."""
    return float(np.mean(np.linalg.norm(a - b, axis=1)))

def min_hand_dist_tolerant(a, b):
    """
    Computes minimum Euclidean distance between hand 'a' and hand 'b',
    testing both direct orientation and horizontal mirror flip (x -> -x).
    """
    d_direct = hand_dist(a, b)
    a_flipped = a.copy()
    a_flipped[:, 0] = -a_flipped[:, 0]
    d_flipped = hand_dist(a_flipped, b)
    return min(d_direct, d_flipped)

def frame_dist(frame_wrist_rel_126, ref_shape_dict):
    """
    Slot-invariant & mirror-tolerant shape distance from notebook Cell 13.
    Compares live hand(s) to reference left/right shapes with tolerance
    for left/right slot assignment, single-handed execution, and mirror inversion.
    """
    u = [
        frame_wrist_rel_126[:63].reshape(21, 3),
        frame_wrist_rel_126[63:].reshape(21, 3)
    ]
    up = [np.count_nonzero(h) > 0 for h in u]

    ref_left = np.array(ref_shape_dict["left"], dtype=np.float32) if ref_shape_dict and ref_shape_dict.get("left") else None
    ref_right = np.array(ref_shape_dict["right"], dtype=np.float32) if ref_shape_dict and ref_shape_dict.get("right") else None
    ref_hands = [ref_left, ref_right]
    ref_present = [ref_left is not None, ref_right is not None]

    if not any(ref_present) or not any(up):
        return 1.0

    # If user has only 1 hand detected: compare to all available non-empty reference hands
    detected_user_hands = [u[i] for i in range(2) if up[i]]
    available_ref_hands = [ref_hands[i] for i in range(2) if ref_present[i]]

    if len(detected_user_hands) == 1:
        user_h = detected_user_hands[0]
        distances = [min_hand_dist_tolerant(user_h, rh) for rh in available_ref_hands]
        return min(distances) if distances else 1.0

    # Both hands detected: evaluate both direct (0->0, 1->1) and cross (0->1, 1->0)
    best = np.inf
    for pr in ((0, 1), (1, 0)):
        ds = [min_hand_dist_tolerant(u[i], ref_hands[pr[i]]) for i in range(2) if up[i] and ref_present[pr[i]]]
        if ds:
            best = min(best, float(np.mean(ds)))

    return best if np.isfinite(best) else 1.0

def calculate_shape_accuracy(frame_wrist_rel_126, ref_shape_dict, scale=0.6166):
    """
    Converts distance to 0-100% score based on median inter-class training distance scale.
    """
    dist = frame_dist(frame_wrist_rel_126, ref_shape_dict)
    score = np.clip(100.0 * (1.0 - (dist / scale)), 0.0, 100.0)
    return round(float(score), 2), dist

def get_finger_feedback(frame_wrist_rel_126, ref_shape_dict):
    """
    Pinpoints which finger has the highest positional discrepancy.
    Tolerant to slot swapping and mirror orientation.
    """
    u = [
        frame_wrist_rel_126[:63].reshape(21, 3),
        frame_wrist_rel_126[63:].reshape(21, 3)
    ]
    live_hand = None
    ref_left = np.array(ref_shape_dict["left"], dtype=np.float32) if ref_shape_dict and ref_shape_dict.get("left") else None
    ref_right = np.array(ref_shape_dict["right"], dtype=np.float32) if ref_shape_dict and ref_shape_dict.get("right") else None

    # Pick dominant detected hand
    if np.count_nonzero(u[1]) >= np.count_nonzero(u[0]) and np.count_nonzero(u[1]) > 0:
        live_hand = u[1]
    elif np.count_nonzero(u[0]) > 0:
        live_hand = u[0]

    if live_hand is None:
        return "Show your hand clearly in front of the camera."

    # Find the closest reference hand and test direct vs mirror
    candidates = []
    for rh in (ref_right, ref_left):
        if rh is not None:
            candidates.append((hand_dist(live_hand, rh), live_hand, rh))
            lh_flip = live_hand.copy()
            lh_flip[:, 0] = -lh_flip[:, 0]
            candidates.append((hand_dist(lh_flip, rh), lh_flip, rh))

    if not candidates:
        return "Match the gesture guide on the right."

    candidates.sort(key=lambda c: c[0])
    _, matched_live, matched_ref = candidates[0]

    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    finger_ranges = [(1, 5), (5, 9), (9, 13), (13, 17), (17, 21)]

    worst_finger = None
    worst_dist = -1.0

    for name, (start, end) in zip(finger_names, finger_ranges):
        dist = float(np.mean(np.linalg.norm(matched_live[start:end] - matched_ref[start:end], axis=1)))
        if dist > worst_dist:
            worst_dist = dist
            worst_finger = name

    if worst_dist < 0.28:
        return "Great form! Your hand positioning matches well."
    return f"Adjust your {worst_finger} finger position."