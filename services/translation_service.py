import socket
import ssl
import time
from config import Config
from services.grammar_service import reconstruct_english_sentence
from services.indic_translator import translate_english_to_indic

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None

# Canonical display name -> ISO-639-1 code
SUPPORTED_LANGUAGES = dict(Config.SUPPORTED_LANGUAGES)

# Case-insensitive lookup supporting both language names and ISO codes
_LANG_LOOKUP = {}
for _name, _code in SUPPORTED_LANGUAGES.items():
    _LANG_LOOKUP[_name.lower()] = (_name, _code)
    _LANG_LOOKUP[_code.lower()] = (_name, _code)

_REACHABILITY_CACHE = {"checked_at": 0.0, "online": False}
_CACHE_TTL_SECONDS = 60.0


def mark_cloud_unreachable() -> None:
    _REACHABILITY_CACHE["checked_at"] = time.monotonic()
    _REACHABILITY_CACHE["online"] = False


def is_google_reachable(timeout: float = 0.75) -> bool:
    """Fast cached TLS handshake check so SNI-filtered proxies fail fast in <15ms once."""
    now = time.monotonic()
    if now - _REACHABILITY_CACHE["checked_at"] < _CACHE_TTL_SECONDS:
        return _REACHABILITY_CACHE["online"]
    online = False
    ctx = ssl.create_default_context()
    try:
        with socket.create_connection(("translate.google.com", 443), timeout=timeout) as raw_sock:
            with ctx.wrap_socket(raw_sock, server_hostname="translate.google.com"):
                online = True
    except OSError:
        online = False
    _REACHABILITY_CACHE["checked_at"] = now
    _REACHABILITY_CACHE["online"] = online
    return online


def resolve_language(target_language: str) -> tuple[str, str]:
    """
    Validate and normalize target language name or ISO code.
    Raises ValueError if target_language is missing or unsupported.
    """
    if not isinstance(target_language, str) or not target_language.strip():
        raise ValueError(
            f"Target language must be a non-empty string. Supported: {', '.join(SUPPORTED_LANGUAGES.keys())}."
        )
    key = target_language.strip().lower()
    if key not in _LANG_LOOKUP:
        raise ValueError(
            f"Unsupported target language '{target_language}'. Supported: {', '.join(SUPPORTED_LANGUAGES.keys())}."
        )
    return _LANG_LOOKUP[key]


def translate_text(text: str, target_language: str = "English") -> str:
    """
    Clean and reconstruct natural English from ISL gloss / user prompt, then translate
    into the validated target language (with resilient online + offline Indic fallback).
    """
    canonical_name, lang_code = resolve_language(target_language)
    english_text = reconstruct_english_sentence(text)

    if lang_code == "en":
        return english_text

    # Tier 1: Cloud NMT via deep-translator when internet access to Google Translate is available
    if GoogleTranslator is not None and is_google_reachable():
        try:
            remote_result = GoogleTranslator(source="en", target=lang_code).translate(english_text)
            if remote_result and remote_result.strip() and remote_result.strip().lower() != english_text.strip().lower():
                return remote_result.strip()
        except Exception as exc:
            mark_cloud_unreachable()
            print(f"[Translation] Cloud translator unavailable ({exc}); using offline Indic grammar engine.")

    # Tier 2: Offline compositional Indic SOV translator + native Unicode script renderer
    return translate_english_to_indic(english_text, lang_code)
