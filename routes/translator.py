import os
import numpy as np
from flask import Blueprint, render_template, request, jsonify, session, send_from_directory
from utils.decorators import login_required
from services.landmark_service import extract_landmarks_from_payload, extract_sequence_from_frames
from services.lstm_service import predict_word
from services.translation_service import translate_text
from services.audio_service import generate_audio
from database import get_connection
from config import Config

translator = Blueprint("translator", __name__)

@translator.route("/translator")
@login_required
def index():
    return render_template("translator.html", supported_languages=Config.SUPPORTED_LANGUAGES)

@translator.route("/api/predict_word", methods=["POST"])
@login_required
def api_predict_word():
    data = request.get_json() or {}
    
    sequence_raw = data.get("sequence", [])
    sequence = [extract_landmarks_from_payload(frame) for frame in sequence_raw] if sequence_raw else []
    
    # Check if sequence has detected hand landmarks
    has_detected_hands = any(np.count_nonzero(f) > 0 for f in sequence) if sequence else False
    
    if not has_detected_hands:
        frames_raw = data.get("frames", [])
        if frames_raw and len(frames_raw) > 0:
            sequence = extract_sequence_from_frames(frames_raw)

    if not sequence or len(sequence) == 0:
        return jsonify({"error": "No frames or sequence provided"}), 400

    predicted_word, confidence, top3 = predict_word(sequence)
    
    return jsonify({
        "predicted_word": predicted_word,
        "confidence": confidence,
        "top3": top3
    })

@translator.route("/api/translate_sentence", methods=["POST"])
@login_required
def api_translate_sentence():
    data = request.get_json() or {}
    sentence = data.get("sentence", "").strip()
    language = data.get("language", session.get("language", "English"))
    
    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400
        
    translated = translate_text(sentence, language)
    audio_file = generate_audio(translated, language)
    
    user_id = session.get("user_id")
    if user_id:
        conn = get_connection()
        conn.execute(
            "INSERT INTO activity_log (user_id, activity) VALUES (?, ?)",
            (user_id, f"Translated: {sentence} -> {translated} ({language})")
        )
        conn.commit()
        conn.close()

    return jsonify({
        "original": sentence,
        "translated": translated,
        "audio_file": audio_file
    })

@translator.route("/static/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(Config.AUDIO_FOLDER, filename)