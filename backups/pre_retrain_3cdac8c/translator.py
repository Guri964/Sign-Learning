import numpy as np
from flask import Blueprint, render_template, request, jsonify, session, send_from_directory
from utils.decorators import login_required
from services.landmark_service import extract_landmarks_from_payload, extract_sequence_from_frames
from services.lstm_service import predict_word
from services.grammar_service import clean_input_text, reconstruct_english_sentence, text_to_isl_gloss
from services.translation_service import resolve_language, translate_text
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
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a valid JSON object."}), 400

    sequence_raw = data.get("sequence")
    frames_raw = data.get("frames")

    if sequence_raw is not None and not isinstance(sequence_raw, list):
        return jsonify({"error": "'sequence' must be a list of frame landmarks."}), 400
    if frames_raw is not None and not isinstance(frames_raw, list):
        return jsonify({"error": "'frames' must be a list of base64 image frames."}), 400

    try:
        sequence = [extract_landmarks_from_payload(frame) for frame in sequence_raw] if sequence_raw else []
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    has_detected_hands = any(np.count_nonzero(f) > 0 for f in sequence) if sequence else False

    if not has_detected_hands and frames_raw:
        try:
            sequence = extract_sequence_from_frames(frames_raw)
        except Exception as exc:
            return jsonify({"error": f"Invalid base64 image frame data: {exc}"}), 400

    if not sequence:
        return jsonify({"error": "No frames or sequence provided"}), 400

    predicted_word, confidence, top3 = predict_word(sequence)

    return jsonify({
        "predicted_word": predicted_word,
        "confidence": confidence,
        "top3": top3,
    })


@translator.route("/api/translate_sentence", methods=["POST"])
@login_required
def api_translate_sentence():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a valid JSON object."}), 400

    # Accept 'sentence' (primary) as well as 'text' / 'prompt' aliases
    raw_input = None
    for key in ("sentence", "text", "prompt"):
        if key in data and data[key] is not None:
            raw_input = data[key]
            break

    if raw_input is None:
        return jsonify({"error": "No sentence provided"}), 400

    raw_language = data.get("language")
    if raw_language is None:
        raw_language = session.get("language", "English")

    try:
        cleaned_original = clean_input_text(raw_input)
        canonical_language, lang_code = resolve_language(raw_language)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    english_sentence = reconstruct_english_sentence(cleaned_original)
    isl_result = text_to_isl_gloss(english_sentence)
    translated = translate_text(cleaned_original, canonical_language)
    audio_file = generate_audio(translated, canonical_language)

    user_id = session.get("user_id")
    if user_id:
        conn = get_connection()
        try:
            conn.execute(
                "INSERT INTO activity_log (user_id, activity) VALUES (?, ?)",
                (user_id, f"Translated: {cleaned_original} -> {translated} ({canonical_language})"),
            )
            conn.commit()
        finally:
            conn.close()

    return jsonify({
        "original": cleaned_original,
        "english_sentence": english_sentence,
        "translated": translated,
        "language": canonical_language,
        "language_code": lang_code,
        "isl_gloss": isl_result["isl_gloss"],
        "isl_tokens": isl_result["isl_tokens"],
        "sign_sequence": isl_result["sign_sequence"],
        "audio_file": audio_file,
        "audio_url": f"/static/audio/{audio_file}" if audio_file else None,
    })


@translator.route("/static/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(Config.AUDIO_FOLDER, filename)
