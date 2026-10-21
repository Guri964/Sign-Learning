import os
from flask import Blueprint, render_template, request, jsonify, session
from utils.decorators import login_required
from utils.helpers import GESTURE_HINTS
from services.landmark_service import extract_landmarks_from_payload, extract_sequence_from_frames
from services.mlp_service import predict_and_score, get_reference_landmarks, get_reference_shape, get_all_words
from database import get_connection
from config import Config

tutor = Blueprint("tutor", __name__)

WORD_CATEGORIES = {
    "Greetings & Basics": ['Hello', 'Thank you', 'How are you', 'Good Morning', 'Good night'],
    "Emotions": ['happy', 'good', 'bad', 'Pleased', 'Alright', 'cold'],
    "Family & People": ['Mother', 'Father', 'Brother', 'Sister', 'Son', 'Daughter', 'Grandfather', 'Grandmother', 'Husband', 'Wife'],
    "Professions": ['Doctor', 'Teacher', 'Police', 'Soldier', 'Lawyer', 'Reporter', 'Manager', 'Waiter'],
    "Places": ['School', 'Hospital', 'Market', 'Restaurant', 'Library', 'Temple', 'University', 'Bank'],
    "Animals": ['Dog', 'Cat', 'Horse', 'Bird', 'Fish', 'Cow'],
    "Colors": ['Red', 'Blue', 'Green', 'Yellow', 'Black', 'White'],
    "Objects": ['Book', 'Pen', 'Table', 'Chair', 'Door', 'Window', 'Telephone', 'Computer'],
    "Transport": ['Car', 'Bus', 'Train', 'Bicycle', 'Plane'],
    "Time": ['Morning', 'Afternoon', 'Evening', 'Today', 'Tomorrow'],
    "Pronouns": ['I', 'you', 'we']
}

@tutor.route("/tutor")
@login_required
def index():
    words = get_all_words()
    return render_template("tutor.html", words=words, categories=WORD_CATEGORIES)

@tutor.route("/gestures")
@login_required
def gestures():
    words = get_all_words()
    gestures_list = []
    for w in words:
        gif_filename = f"{w.replace(' ', '_').replace('/', '_')}.gif"
        gif_path = os.path.join(Config.GESTURE_GIFS_DIR, gif_filename)
        has_gif = os.path.exists(gif_path)
        gif_url = f"/static/gesture_gifs/{gif_filename}" if has_gif else None
        
        # Categorize
        cat = "General"
        for cname, cwords in WORD_CATEGORIES.items():
            if w in cwords:
                cat = cname
                break
                
        hint = GESTURE_HINTS.get(w, "")
        steps = [s.strip() for s in hint.split("|")] if hint else []
        is_both_hands = "both hands" in hint.lower() or "both" in hint.lower()
        
        gestures_list.append({
            "word": w,
            "category": cat,
            "instructions": hint,
            "steps": steps,
            "gif_url": gif_url,
            "hands": "Both Hands" if is_both_hands else "Single Hand"
        })
    return render_template("gestures.html", gestures=gestures_list, categories=WORD_CATEGORIES)

@tutor.route("/api/get_reference", methods=["POST"])
@login_required
def api_get_reference():
    data = request.get_json() or {}
    word = data.get("word", "").strip()
    ref = get_reference_landmarks(word)
    shape = get_reference_shape(word)
    
    gif_filename = f"{word.replace(' ', '_').replace('/', '_')}.gif"
    gif_path = os.path.join(Config.GESTURE_GIFS_DIR, gif_filename)
    has_gif = os.path.exists(gif_path)
    gif_url = f"/static/gesture_gifs/{gif_filename}" if has_gif else None
    instructions = GESTURE_HINTS.get(word, "Perform the sign clearly facing the camera.")

    return jsonify({
        "word": word,
        "reference_landmarks": ref,
        "reference_shape": shape,
        "gif_url": gif_url,
        "instructions": instructions
    })

@tutor.route("/api/score_attempt", methods=["POST"])
@login_required
def api_score_attempt():
    data = request.get_json() or {}
    selected_word = data.get("selected_word", "").strip()
    
    if not selected_word:
        return jsonify({"error": "No word selected"}), 400

    # Prioritize clean live hand data from browser MediaPipe
    hand_data = data.get("hand_data", {})
    if hand_data and (hand_data.get("left_hand") or hand_data.get("right_hand")):
        landmarks_input = extract_landmarks_from_payload(hand_data)
    else:
        # Fallback to frames if no direct hand landmarks were captured
        frames_raw = data.get("frames", [])
        if frames_raw and len(frames_raw) > 0:
            landmarks_input = extract_sequence_from_frames(frames_raw)
        else:
            landmarks_input = extract_landmarks_from_payload(hand_data)

    result = predict_and_score(landmarks_input, selected_word)
    user_id = session.get("user_id")

    if user_id:
        conn = get_connection()
        conn.execute(
            "INSERT INTO progress (user_id, word, accuracy, module) VALUES (?, ?, ?, ?)",
            (user_id, selected_word, result["accuracy"], "tutor")
        )
        if result["badge_unlocked"]:
            badge_title = f"Master: {selected_word}"
            existing = conn.execute(
                "SELECT id FROM badges WHERE user_id = ? AND badge_name = ?",
                (user_id, badge_title)
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO badges (user_id, badge_name) VALUES (?, ?)",
                    (user_id, badge_title)
                )
        conn.execute(
            "INSERT INTO activity_log (user_id, activity) VALUES (?, ?)",
            (user_id, f"Practiced: {selected_word} — {result['accuracy']}%")
        )
        conn.commit()
        conn.close()

    return jsonify(result)

@tutor.route("/api/leaderboard", methods=["GET"])
@login_required
def api_leaderboard():
    conn = get_connection()
    rows = conn.execute("""
        SELECT u.full_name, COUNT(DISTINCT p.word) as signs, AVG(p.accuracy) as avg_acc
        FROM progress p JOIN users u ON p.user_id = u.id
        WHERE p.module = 'tutor'
        GROUP BY p.user_id
        ORDER BY avg_acc DESC
        LIMIT 10
    """).fetchall()
    conn.close()
    leaderboard = [{"name": r["full_name"], "signs": r["signs"], "avg_accuracy": round(r["avg_acc"], 2)} for r in rows]
    return jsonify(leaderboard)

@tutor.route("/api/accuracy_trend", methods=["GET"])
@login_required
def api_accuracy_trend():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify([])
    conn = get_connection()
    rows = conn.execute("""
        SELECT accuracy, practiced_at
        FROM progress
        WHERE user_id = ?
        ORDER BY practiced_at ASC
        LIMIT 15
    """, (user_id,)).fetchall()
    conn.close()
    trend = [{"accuracy": r["accuracy"], "date": r["practiced_at"][:10]} for r in rows]
    return jsonify(trend)