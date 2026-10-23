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