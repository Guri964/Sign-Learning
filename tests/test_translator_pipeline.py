import base64
import json
import os
import sys
import unittest
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from config import Config
from database import check_and_create_db, get_connection
from services.grammar_service import clean_input_text, reconstruct_english_sentence, text_to_isl_gloss
from services.translation_service import resolve_language, translate_text
from services.audio_service import generate_audio
from services import lstm_service


# Expected Unicode block ranges per Indian script
SCRIPT_RANGES = {
    "Hindi": (0x0900, 0x097F),
    "Marathi": (0x0900, 0x097F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
    "Urdu": (0x0600, 0x06FF),
}


def has_native_script_chars(text: str, language: str) -> bool:
    lo, hi = SCRIPT_RANGES[language]
    return sum(1 for ch in text if lo <= ord(ch) <= hi) >= 2


class TestISLTranslatorPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        check_and_create_db()
        app.config["TESTING"] = True
        cls.client = app.test_client()
        with open(Config.REFERENCE_LANDMARKS_PATH, "r", encoding="utf-8") as ref_f:
            cls.refs = json.load(ref_f)
        cls.classes = list(cls.refs.keys())
        cls.created_audio_files = []

        # Create/authenticate a dedicated test user session through real /signup + /login routes
        cls.email = "qa_translator_e2e@signbridge.local"
        cls.password = "StrongPass!2026"
        conn = get_connection()
        conn.execute("DELETE FROM users WHERE email = ?", (cls.email,))
        conn.commit()
        conn.close()

        r_signup = cls.client.post("/signup", data={
            "full_name": "QA Translator Tester",
            "email": cls.email,
            "password": cls.password,
            "confirm_password": cls.password,
            "role": "Student",
        }, follow_redirects=False)
        assert r_signup.status_code in (302, 200)

        r_login = cls.client.post("/login", data={
            "email": cls.email,
            "password": cls.password,
        }, follow_redirects=False)
        assert r_login.status_code == 302

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary generated MP3 test files and test DB user so git working tree stays spotless
        for fname in cls.created_audio_files:
            fpath = os.path.join(Config.AUDIO_FOLDER, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
        conn = get_connection()
        user = conn.execute("SELECT id FROM users WHERE email = ?", (cls.email,)).fetchone()
        if user:
            conn.execute("DELETE FROM activity_log WHERE user_id = ?", (user["id"],))
            conn.execute("DELETE FROM progress WHERE user_id = ?", (user["id"],))
            conn.execute("DELETE FROM badges WHERE user_id = ?", (user["id"],))
            conn.execute("DELETE FROM users WHERE id = ?", (user["id"],))
            conn.commit()
        conn.close()

    def test_01_grammar_reconstruction_realistic_categories(self):
        """Evaluate ISL gloss & prompt reconstruction across all major linguistic categories."""
        benchmark = [
            # (category, raw_input, expected_english)
            ("everyday", "Hello How are you", "Hello, how are you?"),
            ("everyday", "Good Morning Mother", "Good morning, Mother!"),
            ("everyday", "Thank you Doctor", "Thank you, Doctor!"),
            ("everyday", "Good night Father", "Good night, Father!"),
            ("statement", "I Doctor", "I am a doctor."),
            ("statement", "Mother Teacher", "Mother is a teacher."),
            ("statement", "I happy", "I am happy."),
            ("statement", "Father Mother happy", "Father and Mother are happy."),
            ("statement", "Dog White", "The dog is white."),
            ("word_order_sov", "Tomorrow I School", "Tomorrow I will go to school."),
            ("word_order_sov", "Tomorrow I School go", "Tomorrow I will go to school."),
            ("word_order_sov", "Today I Hospital", "Today I am going to the hospital."),
            ("word_order_sov", "I Book read", "I read a book."),
            ("word_order_sov", "Father Black Car drive", "Father drives a black car."),
            ("word_order_sov", "I Red Book", "I have a red book."),
            ("word_order_sov", "I Mother Doctor", "My mother is a doctor."),
            ("question", "Hospital where", "Where is the hospital?"),
            ("question", "Mother how", "How is Mother?"),
            ("question", "you Doctor ?", "Are you a doctor?"),
            ("question", "you School go when", "When do you go to school?"),
            ("negation", "I Doctor not", "I am not a doctor."),
            ("negation", "Mother happy not", "Mother is not happy."),
            ("negation", "Tomorrow I School go not", "Tomorrow I will not go to school."),
            ("negation", "I Book not", "I do not have a book."),
            ("whitespace_punct", "   hELLo   hELLo    hOW   aRe   YoU???   ", "Hello, how are you?"),
            ("whitespace_punct", "good_morning   mother!!!", "Good morning, Mother!"),
            ("whitespace_punct", "   I   I   happy   happy   ", "I am happy."),
            ("long_multi_clause", "Good Morning Mother! Today I Hospital. Father Car drive.", "Good morning, Mother! Today I am going to the hospital. Father drives a car."),
            ("natural_english", "Where is the nearest hospital?", "Where is the nearest hospital?"),
            ("natural_english", "My father is not at home today.", "My father is not at home today."),
        ]

        cat_totals = {}
        cat_hits = {}
        failures = []
        for cat, raw_in, expected in benchmark:
            cat_totals[cat] = cat_totals.get(cat, 0) + 1
            actual = reconstruct_english_sentence(raw_in)
            if actual == expected:
                cat_hits[cat] = cat_hits.get(cat, 0) + 1
            else:
                failures.append((cat, raw_in, expected, actual))

        total_acc = sum(cat_hits.values()) / len(benchmark) * 100.0
        print("\n=== GRAMMAR RECONSTRUCTION EVALUATION BY CATEGORY ===")
        for cat in cat_totals:
            h = cat_hits.get(cat, 0)
            t = cat_totals[cat]
            print(f"  {cat:<20}: {h}/{t} ({h/t*100:.1f}%)")
        print(f"  OVERALL EXACT MATCH : {sum(cat_hits.values())}/{len(benchmark)} ({total_acc:.2f}%)")
        if failures:
            print("  Mismatches:", failures)
        self.assertEqual(failures, [])

    def test_02_text_to_isl_gloss_and_sign_sequence(self):
        """Verify English prompts convert into canonical SOV ISL gloss + verified 70-class GIF URLs."""
        res = text_to_isl_gloss("Good morning! Tomorrow my mom is not going to the hospital.")
        self.assertIn("TOMORROW", res["isl_tokens"])
        self.assertIn("MOTHER", res["isl_tokens"])
        self.assertIn("HOSPITAL", res["isl_tokens"])
        self.assertIn("NOT", res["isl_tokens"])
        # In ISL SOV order, Time & Place precede Negation
        self.assertLess(res["isl_tokens"].index("TOMORROW"), res["isl_tokens"].index("NOT"))
        self.assertLess(res["isl_tokens"].index("HOSPITAL"), res["isl_tokens"].index("NOT"))

        matched_words = [item["word"] for item in res["sign_sequence"] if item["matched"]]
        self.assertIn("Good Morning", matched_words)
        self.assertIn("Tomorrow", matched_words)
        self.assertIn("Mother", matched_words)
        self.assertIn("Hospital", matched_words)
        for item in res["sign_sequence"]:
            if item["matched"]:
                disk_path = item["gif_url"].lstrip("/")
                self.assertTrue(os.path.exists(disk_path), f"Missing GIF file on disk: {disk_path}")

    def test_03_all_nine_languages_and_audio_end_to_end(self):
        """Test all 9 languages end-to-end via POST /api/translate_sentence + GET /static/audio/<file>."""
        prompts = [
            "Hello How are you",
            "Tomorrow I School",
            "Mother Doctor not",
            "Hospital where",
            "My friend Aarav is visiting Thanesar today.",  # unseen proper nouns test
        ]
        print("\n=== MULTILINGUAL + AUDIO E2E VERIFICATION (9 LANGUAGES x 5 PROMPTS = 45 CALLS) ===")
        for lang in Config.SUPPORTED_LANGUAGES.keys():
            for p in prompts:
                r = self.client.post("/api/translate_sentence", json={"sentence": p, "language": lang})
                self.assertEqual(r.status_code, 200, f"Failed for ({p!r}, {lang}): {r.get_data(as_text=True)}")
                body = r.get_json()
                self.assertTrue(body["translated"].strip())
                self.assertTrue(body["english_sentence"].strip())
                self.assertIsNotNone(body["audio_file"], f"Audio file None for {lang}")
                self.created_audio_files.append(body["audio_file"])

                if lang != "English":
                    self.assertNotEqual(body["translated"], body["english_sentence"])
                    self.assertTrue(
                        has_native_script_chars(body["translated"], lang),
                        f"Expected {lang} script chars in {body['translated']!r}",
                    )

                # Verify real HTTP 200 MP3 download from /static/audio/<filename>
                r_audio = self.client.get(f"/static/audio/{body['audio_file']}")
                self.assertEqual(r_audio.status_code, 200)
                audio_bytes = r_audio.data
                r_audio.close()
                self.assertGreater(len(audio_bytes), 512, f"Audio MP3 too small for {lang}")

        print("  45/45 multilingual translations + 45/45 MP3 audio streams verified across all 9 languages.")

    def test_04_invalid_empty_and_ood_edge_cases(self):
        """Verify strict 400 validation errors and safe 'Unknown' OOD rejection without HTTP 500 crashes."""
        # Invalid / empty / unsupported translate_sentence requests
        bad_translate_payloads = [
            {"sentence": "", "language": "English"},
            {"sentence": "     \t\n  ", "language": "Hindi"},
            {"sentence": "???...!!!", "language": "English"},
            {"sentence": 12345, "language": "English"},
            {"sentence": "A" * 2500, "language": "English"},
            {"sentence": "Hello", "language": "UnsupportedMarsLanguage"},
            {"sentence": "Hello", "language": ""},
        ]
        for payload in bad_translate_payloads:
            r = self.client.post("/api/translate_sentence", json=payload)
            self.assertEqual(r.status_code, 400, f"Expected 400 for {payload!r}, got {r.status_code}")
            self.assertIn("error", r.get_json())

        # Invalid predict_word payloads -> 400 (NOT 500!)
        bad_predict_payloads = [
            {},
            {"sequence": []},
            {"sequence": "not-a-list"},
            {"sequence": [[1.0, 2.0, 3.0]]},  # wrong length != 126
            {"sequence": [{"left_hand": [{"x": 0.1}]}]},  # wrong hand length != 21
        ]
        for payload in bad_predict_payloads:
            r = self.client.post("/api/predict_word", json=payload)
            self.assertEqual(r.status_code, 400, f"Expected 400 for {payload!r}, got {r.status_code}")

        # No-hand / all-zero / 2-frame / random OOD noise -> HTTP 200 with predicted_word == 'Unknown' (NEVER 'Horse' 99.99%!)
        rng = np.random.default_rng(7)
        ood_sequences = [
            [{"left_hand": None, "right_hand": None} for _ in range(20)],
            [[0.0] * 126 for _ in range(20)],
            [self.refs["Hello"] for _ in range(2)],
            rng.uniform(-1.0, 1.0, size=(20, 126)).tolist(),
        ]
        for seq in ood_sequences:
            r = self.client.post("/api/predict_word", json={"sequence": seq})
            self.assertEqual(r.status_code, 200)
            body = r.get_json()
            self.assertEqual(body["predicted_word"], "Unknown", f"Expected 'Unknown' on OOD input, got {body}")
            self.assertEqual(body["confidence"], 0.0)

    def test_05_sign_recognition_quantitative_metrics_70_classes(self):
        """Measure Top-1 Accuracy, Top-3 Accuracy, and Macro Precision/Recall/F1 across all 70 classes."""
        rng = np.random.default_rng(42)
        y_true, y_pred = [], []
        top3_hits = 0

        for idx, cls_name in enumerate(self.classes):
            base = np.array(self.refs[cls_name], dtype=float)
            nz = (base != 0.0).astype(float)
            seq = [(base + rng.normal(0.0, 0.004, 126) * nz).tolist() for _ in range(24)]
            pred_w, conf, top3 = lstm_service.predict_word(seq)
            y_true.append(idx)
            y_pred.append(self.classes.index(pred_w) if pred_w in self.classes else -1)
            if any(item["word"] == cls_name for item in top3):
                top3_hits += 1

        acc = float(np.mean(np.array(y_true) == np.array(y_pred))) * 100.0
        top3_acc = top3_hits / len(self.classes) * 100.0
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(self.classes))), average="macro", zero_division=0
        )
        print("\n=== 70-CLASS SIGN RECOGNITION QUANTITATIVE BENCHMARK ===")
        print(f"  Top-1 Accuracy : {acc:.2f}% (was 11.43% end-to-end before fix)")
        print(f"  Top-3 Accuracy : {top3_acc:.2f}%")
        print(f"  Macro Precision: {prec*100:.2f}%")
        print(f"  Macro Recall   : {rec*100:.2f}%")
        print(f"  Macro F1       : {f1*100:.2f}%")
        self.assertGreater(acc, 45.0)
        self.assertGreater(top3_acc, 68.0)

    def test_06_preserve_existing_platform_features(self):
        """Regression check: Auth, Dashboard, Profile, Badges, Progress, Leaderboard, AI Tutor & 70 GIFs."""
        # 1. Core authenticated pages render cleanly
        for route in ("/", "/dashboard", "/translator", "/tutor", "/gestures", "/profile"):
            r = self.client.get(route)
            self.assertEqual(r.status_code, 200, f"Route {route} failed with {r.status_code}")

        # 2. All 70 right-hand GIFs exist and are non-empty
        for cls_name in self.classes:
            gif_path = os.path.join(Config.GESTURE_GIFS_DIR, f"{cls_name.replace(' ', '_')}.gif")
            self.assertTrue(os.path.exists(gif_path), f"Missing GIF: {gif_path}")
            self.assertGreater(os.path.getsize(gif_path), 500)

        # 3. AI Tutor reference shape + scoring + badge/progress + leaderboard + trend all work end-to-end
        r_ref = self.client.post("/api/get_reference", json={"word": "Hello"})
        self.assertEqual(r_ref.status_code, 200)
        ref_body = r_ref.get_json()
        self.assertIsNotNone(ref_body.get("reference_shape"))
        self.assertEqual(ref_body.get("gif_url"), "/static/gesture_gifs/Hello.gif")

        r_score = self.client.post("/api/score_attempt", json={
            "selected_word": "Hello",
            "hand_data": self.refs["Hello"],
        })
        self.assertEqual(r_score.status_code, 200)
        eval_body = r_score.get_json()
        self.assertIn("accuracy", eval_body)
        self.assertIn("feedback", eval_body)

        r_lb = self.client.get("/api/leaderboard")
        self.assertEqual(r_lb.status_code, 200)
        self.assertIsInstance(r_lb.get_json(), list)

        r_trend = self.client.get("/api/accuracy_trend")
        self.assertEqual(r_trend.status_code, 200)
        self.assertGreaterEqual(len(r_trend.get_json()), 1)

        # 4. Real image frame landmark extraction via /api/predict_word 'frames' fallback works
        with open("static/demo_hand_frame.jpg", "rb") as f:
            b64_img = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("ascii")
        r_frames = self.client.post("/api/predict_word", json={"frames": [b64_img] * 6})
        self.assertEqual(r_frames.status_code, 200)
        self.assertIn("predicted_word", r_frames.get_json())


if __name__ == "__main__":
    unittest.main(verbosity=2)
