import ctypes
import os
import threading
import uuid
from config import Config
from services.translation_service import is_google_reachable, mark_cloud_unreachable, resolve_language

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

try:
    import espeakng_loader
    import lameenc
    _HAS_OFFLINE_TTS = True
except ImportError:
    espeakng_loader = None
    lameenc = None
    _HAS_OFFLINE_TTS = False

_ESPEAK_LOCK = threading.Lock()
_ESPEAK_LIB = None
_ESPEAK_SAMPLE_RATE = 22050
_PCM_CHUNKS: list[bytes] = []
_SYNTH_CB_REF = None

SUPPORTED_TTS_VOICE_MAP = {
    "en": b"en",
    "hi": b"hi",
    "kn": b"kn",
    "ta": b"ta",
    "te": b"te",
    "ml": b"ml",
    "mr": b"mr",
    "ur": b"ur",
    "gu": b"gu",
}


def _ensure_espeak_initialized():
    global _ESPEAK_LIB, _ESPEAK_SAMPLE_RATE, _SYNTH_CB_REF
    if _ESPEAK_LIB is not None or not _HAS_OFFLINE_TTS:
        return

    lib = ctypes.CDLL(espeakng_loader.get_library_path())
    audio_output_synchronous = 0x02
    sr = lib.espeak_Initialize(
        audio_output_synchronous,
        0,
        espeakng_loader.get_data_path().encode("utf-8"),
        0,
    )
    if sr > 0:
        _ESPEAK_SAMPLE_RATE = sr

    synth_cb_type = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_short),
        ctypes.c_int,
        ctypes.c_void_p,
    )

    @synth_cb_type
    def _callback(wav_ptr, num_samples, _events):
        if wav_ptr and num_samples > 0:
            _PCM_CHUNKS.append(ctypes.string_at(wav_ptr, num_samples * 2))
        return 0

    _SYNTH_CB_REF = _callback  # Prevent garbage collection of C callback pointer
    lib.espeak_SetSynthCallback(_SYNTH_CB_REF)
    _ESPEAK_LIB = lib


def _synthesize_offline_mp3(text: str, lang_code: str, filepath: str) -> bool:
    if not _HAS_OFFLINE_TTS:
        return False
    with _ESPEAK_LOCK:
        _ensure_espeak_initialized()
        if _ESPEAK_LIB is None:
            return False
        _PCM_CHUNKS.clear()
        voice = SUPPORTED_TTS_VOICE_MAP.get(lang_code, b"en")
        _ESPEAK_LIB.espeak_SetVoiceByName(voice)
        encoded_text = text.encode("utf-8")
        # espeakCHARS_UTF8 = 1
        _ESPEAK_LIB.espeak_Synth(encoded_text, len(encoded_text) + 1, 0, 0, 0, 1, None, None)
        pcm_bytes = b"".join(_PCM_CHUNKS)
        if not pcm_bytes:
            return False
        encoder = lameenc.Encoder()
        encoder.set_bit_rate(64)
        encoder.set_in_sample_rate(_ESPEAK_SAMPLE_RATE)
        encoder.set_channels(1)
        encoder.set_quality(5)
        mp3_bytes = encoder.encode(pcm_bytes) + encoder.flush()
        if not mp3_bytes:
            return False
        with open(filepath, "wb") as f:
            f.write(mp3_bytes)
    return os.path.exists(filepath) and os.path.getsize(filepath) > 128


def generate_audio(text: str, language: str = "English") -> str | None:
    """
    Synthesize spoken MP3 audio for `text` in `language` (accepts either display
    name e.g. 'Hindi' or ISO code e.g. 'hi') under `Config.AUDIO_FOLDER`.
    Uses cloud gTTS when reachable, and seamlessly falls back to thread-safe offline
    multilingual `espeak-ng` + LAME MP3 synthesis when offline.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    _, lang_code = resolve_language(language)
    os.makedirs(Config.AUDIO_FOLDER, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(Config.AUDIO_FOLDER, filename)

    if gTTS is not None and is_google_reachable():
        try:
            tts = gTTS(text=text.strip(), lang=lang_code)
            tts.save(filepath)
            if os.path.exists(filepath) and os.path.getsize(filepath) > 128:
                return filename
        except Exception as exc:
            mark_cloud_unreachable()
            print(f"[Audio] gTTS cloud synthesis unavailable ({exc}); falling back to offline MP3 synthesis.")

    try:
        if _synthesize_offline_mp3(text.strip(), lang_code, filepath):
            return filename
    except Exception as exc:
        print(f"[Audio] Offline speech synthesis error: {exc}")

    return None
