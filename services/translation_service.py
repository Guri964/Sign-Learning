from googletrans import Translator
from config import Config

translator = Translator()

def translate_text(text, target_language):
    lang_code = Config.SUPPORTED_LANGUAGES.get(target_language, "en")
    if lang_code == "en":
        return text
    try:
        result = translator.translate(text, dest=lang_code)
        return result.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text