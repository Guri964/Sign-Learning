from flask import Blueprint, render_template, session, redirect, url_for, request, jsonify
from database import get_connection
from utils.decorators import login_required

dashboard = Blueprint("dashboard", __name__)

@dashboard.route("/dashboard")
@login_required
def index():
    user_id = session["user_id"]
    conn = get_connection()
    signs_learned = conn.execute(
        "SELECT COUNT(DISTINCT word) FROM progress WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    avg_accuracy = conn.execute(
        "SELECT AVG(accuracy) FROM progress WHERE user_id = ?", (user_id,)
    ).fetchone()[0] or 0.0
    badges = conn.execute(
        "SELECT COUNT(*) FROM badges WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    recent_activity = conn.execute(
        "SELECT activity, logged_at FROM activity_log WHERE user_id = ? ORDER BY logged_at DESC LIMIT 5",
        (user_id,)
    ).fetchall()
    conn.close()
    return render_template("dashboard.html",
        signs_learned=signs_learned,
        avg_accuracy=round(avg_accuracy, 2),
        badges=badges,
        recent_activity=recent_activity
    )

@dashboard.route("/profile")
@login_required
def profile():
    user_id = session["user_id"]
    conn = get_connection()
    progress_data = conn.execute(
        "SELECT word, accuracy, module, practiced_at FROM progress WHERE user_id = ? ORDER BY practiced_at DESC",
        (user_id,)
    ).fetchall()
    badges = conn.execute(
        "SELECT badge_name, earned_at FROM badges WHERE user_id = ?", (user_id,)
    ).fetchall()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return render_template("profile.html",
        progress_data=progress_data,
        badges=badges,
        user=user
    )

@dashboard.route("/update_language", methods=["POST"])
@login_required
def update_language():
    language = request.json.get("language")
    from config import Config
    if language not in Config.SUPPORTED_LANGUAGES:
        return jsonify({"error": "Invalid language"}), 400
    user_id = session["user_id"]
    conn = get_connection()
    conn.execute("UPDATE users SET language_preference = ? WHERE id = ?", (language, user_id))
    conn.commit()
    conn.close()
    session["language"] = language
    return jsonify({"success": True})