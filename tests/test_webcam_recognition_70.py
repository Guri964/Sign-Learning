import hashlib
import json
import os
import uuid
import numpy as np
import pytest
from PIL import Image

from app import app
from config import Config
from database import check_and_create_db, get_connection
from services import lstm_service, mlp_service
from services.grammar_service import text_to_isl_gloss

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "heldout_webcam_validation.npz")


@pytest.fixture(scope="module")
def client():
    check_and_create_db()
    app.config["TESTING"] = True
    email = f"qa_webcam_{uuid.uuid4().hex[:8]}@signbridge.local"
    password = "TestPassword123!"
    with app.test_client() as test_client:
        r_signup = test_client.post(
            "/signup",
            data={
                "full_name": "QA Webcam Verifier",
                "email": email,
                "password": password,
                "confirm_password": password,
                "role": "Student",
            },
            follow_redirects=False,
        )
        assert r_signup.status_code in (200, 302)
        r_login = test_client.post(
            "/login",
            data={"email": email, "password": password},
            follow_redirects=False,
        )
        assert r_login.status_code == 302
        yield test_client

    conn = get_connection()
    user = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    if user:
        conn.execute("DELETE FROM progress WHERE user_id = ?", (user["id"],))
        conn.execute("DELETE FROM badges WHERE user_id = ?", (user["id"],))
        conn.execute("DELETE FROM activity_log WHERE user_id = ?", (user["id"],))
        conn.execute("DELETE FROM users WHERE id = ?", (user["id"],))
        conn.commit()
    conn.close()


@pytest.fixture(scope="module")
def heldout_data():
    assert os.path.exists(FIXTURE_PATH), f"Missing validation fixture at {FIXTURE_PATH}"
    return np.load(FIXTURE_PATH, allow_pickle=True)


def _perturb_webcam_attempt(raw_seq: np.ndarray, scale: float, tx: float, ty: float, mirror: bool = False) -> np.ndarray:
    seq = np.array(raw_seq, dtype=np.float32).copy()
    for base in (0, 63):
        sub = seq[:, base:base + 63].reshape(len(seq), 21, 3)
        mask = np.any(sub != 0.0, axis=(1, 2))
        if np.any(mask):
            sub[mask, :, 0] = 0.50 + (sub[mask, :, 0] - 0.50) * scale + tx
            sub[mask, :, 1] = 0.42 + (sub[mask, :, 1] - 0.42) * scale + ty
            sub[mask, :, 2] *= scale
        seq[:, base:base + 63] = sub.reshape(len(seq), 63)

    if mirror:
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
    return seq


def test_pixel_to_prediction_real_human_webcam_videos(client, heldout_data):
    """
    Full live webcam pipeline test starting from REAL JPEG video frames decoded through
    MediaPipe HandLandmarker -> aspect-ratio correction -> slot stabilization ->
    ISL_Webcam_Kinematic_71 + Studio ensemble -> HTTP response.
    """
    print("\n=== REAL PIXEL-TO-PREDICTION E2E WEBCAM VIDEO VERIFICATION ===")
    for expected_word, key in [
        ("Hello", "b64_hello_laptop"),
        ("How are you", "b64_howareyou_laptop"),
        ("happy", "b64_happy_full"),
        ("happy", "b64_happy_laptop"),
    ]:
        frames_b64 = heldout_data[key].tolist()
        res = client.post("/api/predict_word", json={"frames": frames_b64})
        assert res.status_code == 200, res.get_data(as_text=True)
        body = res.get_json()
        print(f"  [{key}] -> predicted='{body['predicted_word']}' ({body['confidence']:.2f}%)")
        assert body["predicted_word"] == expected_word, (
            f"Expected {expected_word} on real webcam video {key}, got {body}"
        )
        assert body["confidence"] >= 40.0

    # Real human non-vocabulary ISL video ('loud', outside the 70-class vocabulary) must reject as Unknown!
    ood_res = client.post("/api/predict_word", json={"frames": heldout_data["b64_ood_loud"].tolist()})
    assert ood_res.status_code == 200
    ood_body = ood_res.get_json()
    print(f"  [b64_ood_loud (OOD 'loud')] -> predicted='{ood_body['predicted_word']}' ({ood_body['confidence']:.2f}%)")
    assert ood_body["predicted_word"] == "Unknown", f"Expected Unknown on real OOD video, got {ood_body}"


def test_confused_classes_multi_signer_positions_and_scales(heldout_data):
    """
    Explicitly validates formerly confused classes (`Police`, `Bicycle`, `Doctor`, plus
    `Hello`, `happy`, `Father`, `Mother`) across varied webcam scales, positions, and left-handed mirroring.
    """
    classes = [str(c) for c in heldout_data["classes"]]
    seqs = heldout_data["heldout_70_seqs"].astype(np.float32)

    attempt_variants = [
        (1.0, 0.0, 0.0, False),
        (0.80, -0.09, 0.05, False),
        (1.28, 0.08, -0.04, False),
        (1.10, -0.05, 0.02, True),
    ]

    print("\n=== FORMERLY CONFUSED CLASSES MULTI-POSITION / SCALE / CHIRALITY CHECK ===")
    for target_cls in ["Police", "Bicycle", "Doctor", "Hello", "happy", "Father", "Mother"]:
        idx = classes.index(target_cls)
        base = seqs[idx]
        hits = 0
        for scale, tx, ty, mirror in attempt_variants:
            pert = _perturb_webcam_attempt(base, scale, tx, ty, mirror=mirror)
            pred_w, _, top3 = lstm_service.predict_word(pert)
            top3_words = [d["word"] for d in top3]
            if pred_w == target_cls:
                hits += 1
            assert target_cls in top3_words, (
                f"{target_cls} missing from Top-3 {top3_words} under variant {(scale, tx, ty, mirror)}"
            )
        print(f"  {target_cls:10s}: {hits}/4 Top-1 hits across normal, far (0.8x), close (1.28x), and left-handed mirror")
        assert hits >= 3, f"{target_cls} matched Top-1 on only {hits}/4 webcam variants"


def test_heldout_70_classes_multi_attempt_accuracy(heldout_data):
    """
    Tests ALL 167 strictly held-out unseen human recordings across all 70 signs x 3 distinct
    simulated webcam positions/scales (501 live end-to-end `predict_word` calls including open-set gate).
    """
    classes = [str(c) for c in heldout_data["classes"]]
    all_seqs = heldout_data["heldout_all_seqs"].astype(np.float32)
    all_y = heldout_data["heldout_all_y"].astype(np.int64)
    variants = [(1.0, 0.0, 0.0), (0.82, -0.07, 0.04), (1.22, 0.07, -0.03)]

    top1_ok = 0
    top3_ok = 0
    total = 0
    for seq_i, cls_idx in zip(all_seqs, all_y):
        cls_name = classes[int(cls_idx)]
        for sc, tx, ty in variants:
            pert = _perturb_webcam_attempt(seq_i, sc, tx, ty, mirror=False)
            pred_w, _, top3 = lstm_service.predict_word(pert)
            if pred_w == cls_name:
                top1_ok += 1
            if any(d["word"] == cls_name for d in top3):
                top3_ok += 1
            total += 1

    top1_pct = 100.0 * top1_ok / total
    top3_pct = 100.0 * top3_ok / total
    print(f"\n=== STRICT HELD-OUT MULTI-ATTEMPT LIVE GATED BENCHMARK ({total} RUNS ACROSS 70 SIGNS) ===")
    print(f"  Live Gated Top-1: {top1_ok}/{total} ({top1_pct:.2f}%)")
    print(f"  Live Gated Top-3: {top3_ok}/{total} ({top3_pct:.2f}%)")
    assert top1_pct >= 70.0, f"Held-out 501-attempt live gated Top-1 dropped to {top1_pct:.2f}%"
    assert top3_pct >= 78.0, f"Held-out 501-attempt live gated Top-3 dropped to {top3_pct:.2f}%"


def test_no_hand_corrupt_and_real_ood_rejection(heldout_data):
    """
    Verifies empty frames, transient <3 frame flicker, constant/collapsed points,
    random noise, numpy array inputs, and stratified real OOD human signs safely reject to Unknown.
    """
    assert lstm_service.predict_word([]) == ("Unknown", 0.0, [])
    assert lstm_service.predict_word(np.zeros((20, 126), dtype=np.float32))[0] == "Unknown"

    flicker = np.zeros((20, 126), dtype=np.float32)
    flicker[5, 63:126] = 0.5
    flicker[6, 63:126] = 0.5
    assert lstm_service.predict_word(flicker)[0] == "Unknown"

    collapsed = np.full((20, 126), 0.42, dtype=np.float32)
    assert lstm_service.predict_word(collapsed) == ("Unknown", 0.0, [])

    rng = np.random.default_rng(123)
    noise = rng.normal(0.5, 0.25, size=(20, 126)).astype(np.float32)
    assert lstm_service.predict_word(noise) == ("Unknown", 0.0, [])

    ood_seqs = heldout_data["ood_30_seqs"].astype(np.float32)
    unknown_count = sum(1 for s in ood_seqs if lstm_service.predict_word(s)[0] == "Unknown")
    print(f"\n=== REAL NON-VOCABULARY HUMAN OOD REJECTION: {unknown_count}/{len(ood_seqs)} ({100.0*unknown_count/len(ood_seqs):.2f}%) ===")
    assert unknown_count >= int(0.55 * len(ood_seqs)), (
        f"Expected strong Unknown rejection on real OOD human signs, got {unknown_count}/{len(ood_seqs)}"
    )


def test_exact_70_class_label_to_gif_and_ghost_hand_alignment():
    """
    Verifies every index 0..69 aligns identically across label_map.json, label_classes.npy,
    displayed right-hand GIFs in static/gesture_gifs/, and AI Tutor ghost-hand skeletons.
    """
    with open(Config.LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    label_classes = [str(x) for x in np.load(Config.LABEL_CLASSES_PATH, allow_pickle=True)]
    assert len(label_map) == 70
    assert len(label_classes) == 70

    for idx in range(70):
        word = label_classes[idx]
        assert label_map[str(idx)] == word, f"Mismatch at index {idx}: {label_map[str(idx)]} vs {word}"

        gloss_res = text_to_isl_gloss(word)
        assert len(gloss_res["sign_sequence"]) >= 1, f"No GIF returned for class {word}"
        matched = [g for g in gloss_res["sign_sequence"] if g["word"].lower() == word.lower()]
        assert len(matched) == 1, f"Expected exact class GIF match for '{word}', got {gloss_res['sign_sequence']}"

        rel_url = matched[0]["gif_url"].lstrip("/")
        disk_path = os.path.join(Config.BASE_DIR, rel_url)
        expected_stem = word.replace(" ", "_") + ".gif"
        assert os.path.basename(disk_path) == expected_stem
        assert os.path.exists(disk_path), f"Missing GIF file on disk: {disk_path}"
        with Image.open(disk_path) as im:
            assert im.size[0] > 0 and im.size[1] > 0

        ghost = mlp_service.get_target_ghost_hand(word)
        assert ghost is not None and len(ghost) == 21 and len(ghost[0]) == 3, (
            f"Invalid ghost-hand skeleton for {word}"
        )


def test_preserve_all_platform_routes_tutor_auth_dashboard(client, heldout_data):
    """
    Verifies untouched SHA-256 preservation of original studio checkpoints plus full preservation
    of auth, dashboard, gesture GIF library, AI Tutor scoring (empty-frame safety & real attempt
    scoring), leaderboard, accuracy trend, and translator endpoints.
    """
    lstm_sha = hashlib.sha256(open(Config.LSTM_MODEL_PATH, "rb").read()).hexdigest()
    mlp_sha = hashlib.sha256(open(Config.MLP_MODEL_PATH, "rb").read()).hexdigest()
    assert lstm_sha.startswith("fcc42a13"), f"Original studio LSTM checkpoint modified: {lstm_sha}"
    assert mlp_sha.startswith("e0290692"), f"Original studio MLP checkpoint modified: {mlp_sha}"

    for page in ("/dashboard", "/gestures", "/tutor", "/translator", "/api/leaderboard", "/api/accuracy_trend"):
        resp = client.get(page)
        assert resp.status_code == 200, f"GET {page} failed with {resp.status_code}"

    ref = client.post("/api/get_reference", json={"word": "Hello"})
    assert ref.status_code == 200
    ref_body = ref.get_json()
    assert len(ref_body["ghost_hand"]) == 21

    # Empty AI Tutor frame must safely return Unknown & 0% (never hallucinating 'Horse'!)
    empty_score = client.post(
        "/api/score_attempt",
        json={"selected_word": "Hello", "hand_data": {"left_hand": None, "right_hand": None}},
    )
    assert empty_score.status_code == 200
    assert empty_score.get_json()["predicted_word"] == "Unknown"
    assert empty_score.get_json()["accuracy"] == 0.0

    # Real sequence attempt in AI Tutor for 'Police'
    classes = [str(c) for c in heldout_data["classes"]]
    police_seq = heldout_data["heldout_70_seqs"][classes.index("Police")].astype(np.float32).tolist()
    valid_score = client.post(
        "/api/score_attempt",
        json={"selected_word": "Police", "hand_sequence": police_seq},
    )
    assert valid_score.status_code == 200
    vbody = valid_score.get_json()
    assert vbody["predicted_word"] == "Police"
    assert vbody["accuracy"] >= 50.0
