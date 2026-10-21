import os
import uuid
from gtts import gTTS
from config import Config

def generate_audio(text, language):
    lang_code = Config.SUPPORTED_LANGUAGES.get(language, "en")
    audio_folder = Config.AUDIO_FOLDER
    os.makedirs(audio_folder, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(audio_folder, filename)
    try:
        tts = gTTS(text=text, lang=lang_code)
        tts.save(filepath)
        return filename
    except Exception as e:
        print(f"Audio generation error: {e}")
        return None