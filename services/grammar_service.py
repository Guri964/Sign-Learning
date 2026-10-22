import os
import re
import unicodedata
from utils.helpers import GESTURE_HINTS

MAX_PROMPT_LENGTH = 2000

# Canonical 70-sign vocabulary mapped lower-case -> exact dataset label
VOCAB_70 = list(GESTURE_HINTS.keys())
VOCAB_LOWER_MAP = {w.lower(): w for w in VOCAB_70}

# Multi-word vocabulary items & common synonyms -> canonical vocabulary sign
MULTIWORD_PHRASES = [
    ("how are you doing", "How are you"),
    ("how do you do", "How are you"),
    ("how are you", "How are you"),
    ("good morning", "Good Morning"),
    ("good night", "Good night"),
    ("goodnight", "Good night"),
    ("thank you very much", "Thank you"),
    ("thanks a lot", "Thank you"),
    ("thank you", "Thank you"),
]

SYNONYM_TO_SIGN = {
    "hi": "Hello",
    "hey": "Hello",
    "hello": "Hello",
    "namaste": "Hello",
    "greetings": "Hello",
    "thanks": "Thank you",
    "thank": "Thank you",
    "grateful": "Thank you",
    "glad": "Pleased",
    "pleased": "Pleased",
    "delighted": "Pleased",
    "ok": "Alright",
    "okay": "Alright",
    "fine": "Alright",
    "alright": "Alright",
    "joyful": "happy",
    "cheerful": "happy",
    "happy": "happy",
    "well": "good",
    "nice": "good",
    "great": "good",
    "good": "good",
    "unwell": "bad",
    "terrible": "bad",
    "poor": "bad",
    "sad": "bad",
    "bad": "bad",
    "chilly": "cold",
    "freezing": "cold",
    "cold": "cold",
    "mom": "Mother",
    "mum": "Mother",
    "mummy": "Mother",
    "mama": "Mother",
    "mother": "Mother",
    "dad": "Father",
    "daddy": "Father",
    "papa": "Father",
    "father": "Father",
    "bro": "Brother",
    "brother": "Brother",
    "sis": "Sister",
    "sister": "Sister",
    "son": "Son",
    "boy": "Son",
    "daughter": "Daughter",
    "girl": "Daughter",
    "grandpa": "Grandfather",
    "granddad": "Grandfather",
    "grandfather": "Grandfather",
    "grandma": "Grandmother",
    "granny": "Grandmother",
    "grandmother": "Grandmother",
    "husband": "Husband",
    "wife": "Wife",
    "physician": "Doctor",
    "surgeon": "Doctor",
    "doctor": "Doctor",
    "dr": "Doctor",
    "instructor": "Teacher",
    "professor": "Teacher",
    "educator": "Teacher",
    "teacher": "Teacher",
    "policeman": "Police",
    "cop": "Police",
    "officer": "Police",
    "police": "Police",
    "army": "Soldier",
    "warrior": "Soldier",
    "soldier": "Soldier",
    "attorney": "Lawyer",
    "advocate": "Lawyer",
    "lawyer": "Lawyer",
    "journalist": "Reporter",
    "reporter": "Reporter",
    "boss": "Manager",
    "director": "Manager",
    "manager": "Manager",
    "server": "Waiter",
    "waiter": "Waiter",
    "academy": "School",
    "classroom": "School",
    "school": "School",
    "clinic": "Hospital",
    "medical": "Hospital",
    "hospital": "Hospital",
    "bazaar": "Market",
    "shop": "Market",
    "store": "Market",
    "market": "Market",
    "cafe": "Restaurant",
    "hotel": "Restaurant",
    "diner": "Restaurant",
    "restaurant": "Restaurant",
    "library": "Library",
    "shrine": "Temple",
    "church": "Temple",
    "mosque": "Temple",
    "temple": "Temple",
    "college": "University",
    "campus": "University",
    "university": "University",
    "bank": "Bank",
    "puppy": "Dog",
    "hound": "Dog",
    "dog": "Dog",
    "kitten": "Cat",
    "kitty": "Cat",
    "cat": "Cat",
    "pony": "Horse",
    "stallion": "Horse",
    "horse": "Horse",
    "sparrow": "Bird",
    "pigeon": "Bird",
    "bird": "Bird",
    "fish": "Fish",
    "cattle": "Cow",
    "ox": "Cow",
    "cow": "Cow",
    "red": "Red",
    "crimson": "Red",
    "blue": "Blue",
    "green": "Green",
    "yellow": "Yellow",
    "black": "Black",
    "dark": "Black",
    "white": "White",
    "novel": "Book",
    "textbook": "Book",
    "notebook": "Book",
    "book": "Book",
    "pencil": "Pen",
    "marker": "Pen",
    "pen": "Pen",
    "desk": "Table",
    "table": "Table",
    "seat": "Chair",
    "stool": "Chair",
    "chair": "Chair",
    "gate": "Door",
    "entrance": "Door",
    "door": "Door",
    "window": "Window",
    "phone": "Telephone",
    "mobile": "Telephone",
    "cellphone": "Telephone",
    "telephone": "Telephone",
    "laptop": "Computer",
    "pc": "Computer",
    "desktop": "Computer",
    "computer": "Computer",
    "auto": "Car",
    "vehicle": "Car",
    "automobile": "Car",
    "taxi": "Car",
    "cab": "Car",
    "car": "Car",
    "coach": "Bus",
    "bus": "Bus",
    "railway": "Train",
    "metro": "Train",
    "rail": "Train",
    "train": "Train",
    "bike": "Bicycle",
    "cycle": "Bicycle",
    "bicycle": "Bicycle",
    "airplane": "Plane",
    "aeroplane": "Plane",
    "aircraft": "Plane",
    "flight": "Plane",
    "plane": "Plane",
    "morning": "Morning",
    "afternoon": "Afternoon",
    "noon": "Afternoon",
    "evening": "Evening",
    "today": "Today",
    "now": "Today",
    "tomorrow": "Tomorrow",
    "i": "I",
    "me": "I",
    "my": "I",
    "myself": "I",
    "mine": "I",
    "you": "you",
    "your": "you",
    "yours": "you",
    "yourself": "you",
    "we": "we",
    "us": "we",
    "our": "we",
    "ours": "we",
    "ourselves": "we",
}

# Linguistic lexical categories for English clause realization
PRONOUN_SUBJ = {
    "i": "I", "me": "I",
    "you": "you",
    "we": "we", "us": "we",
    "he": "he", "him": "he",
    "she": "she", "her": "she",
    "they": "they", "them": "they",
    "it": "it",
}

PRONOUN_POSS = {
    "i": "my", "me": "my", "my": "my",
    "you": "your", "your": "your",
    "we": "our", "us": "our", "our": "our",
    "he": "his", "him": "his", "his": "his",
    "she": "her", "her": "her",
    "they": "their", "them": "their", "their": "their",
}

KINSHIP_PROPER = {
    "mother": "Mother", "mom": "Mom", "father": "Father", "dad": "Dad",
    "grandfather": "Grandfather", "grandmother": "Grandmother",
}

PERSON_NOUNS = {
    "mother", "mom", "mum", "father", "dad", "brother", "sister",
    "son", "daughter", "grandfather", "grandmother", "husband", "wife",
    "friend", "child", "boy", "girl", "man", "woman", "student", "patient", "person",
}

PROFESSIONS = {
    "doctor", "teacher", "police", "soldier", "lawyer", "reporter",
    "manager", "waiter", "nurse", "engineer", "driver", "farmer", "student",
}

PLACES = {
    "school", "hospital", "market", "restaurant", "library", "temple",
    "university", "bank", "home", "house", "office", "station", "airport",
    "park", "shop", "store", "clinic", "college", "hotel", "class",
}

OBJECTS_OWNABLE = {
    "book", "pen", "table", "chair", "door", "window", "telephone", "phone",
    "computer", "laptop", "car", "bus", "train", "bicycle", "bike", "plane",
    "dog", "cat", "horse", "bird", "fish", "cow", "bag", "water", "food",
    "money", "medicine", "ticket", "key",
}

ADJECTIVES = {
    "happy", "good", "bad", "pleased", "alright", "fine", "okay", "cold",
    "hot", "warm", "sick", "ill", "well", "tired", "hungry", "thirsty",
    "ready", "busy", "free", "late", "early", "big", "small", "new", "old",
    "fast", "slow", "beautiful", "nice", "angry", "sad", "safe",
    "red", "blue", "green", "yellow", "black", "white",
}

COLOR_ADJS = {"red", "blue", "green", "yellow", "black", "white"}

TIME_WORDS = {
    "tomorrow": ("Tomorrow", "future"),
    "yesterday": ("Yesterday", "past"),
    "today": ("Today", "present_prog"),
    "now": ("Now", "present_prog"),
    "morning": ("In the morning,", "present"),
    "afternoon": ("In the afternoon,", "present"),
    "evening": ("In the evening,", "present"),
    "night": ("At night,", "present"),
}

WH_WORDS = {"what", "where", "when", "who", "whom", "whose", "why", "which", "how"}
NEGATORS = {"not", "no", "never", "dont", "don't", "cannot", "can't", "didn't", "didnt", "won't", "wont"}

AUX_AND_FINITE = {
    "am", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "will", "would", "can", "could", "shall", "should",
    "may", "might", "must", "have", "has", "had",
    "am not", "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't", "won't", "can't",
}

FUNCTION_MARKERS = {
    "the", "a", "an", "to", "in", "at", "on", "for", "with", "from", "of",
    "into", "about", "because", "if", "when", "while", "before", "after", "than", "that",
}

# Verb lexicon: lemma -> (3sg_present, prog_participle, simple_past, takes_to_place)
VERBS = {
    "go": ("goes", "going", "went", True),
    "come": ("comes", "coming", "came", True),
    "walk": ("walks", "walking", "walked", True),
    "travel": ("travels", "traveling", "traveled", True),
    "return": ("returns", "returning", "returned", True),
    "drive": ("drives", "driving", "drove", False),
    "ride": ("rides", "riding", "rode", False),
    "take": ("takes", "taking", "took", False),
    "read": ("reads", "reading", "read", False),
    "write": ("writes", "writing", "wrote", False),
    "eat": ("eats", "eating", "ate", False),
    "drink": ("drinks", "drinking", "drank", False),
    "buy": ("buys", "buying", "bought", False),
    "see": ("sees", "seeing", "saw", False),
    "watch": ("watches", "watching", "watched", False),
    "call": ("calls", "calling", "called", False),
    "help": ("helps", "helping", "helped", False),
    "need": ("needs", "needing", "needed", False),
    "want": ("wants", "wanting", "wanted", False),
    "like": ("likes", "liking", "liked", False),
    "love": ("loves", "loving", "loved", False),
    "know": ("knows", "knowing", "knew", False),
    "learn": ("learns", "learning", "learned", False),
    "study": ("studies", "studying", "studied", False),
    "teach": ("teaches", "teaching", "taught", False),
    "work": ("works", "working", "worked", True),
    "live": ("lives", "living", "lived", True),
    "open": ("opens", "opening", "opened", False),
    "close": ("closes", "closing", "closed", False),
    "give": ("gives", "giving", "gave", False),
    "use": ("uses", "using", "used", False),
    "play": ("plays", "playing", "played", False),
    "sleep": ("sleeps", "sleeping", "slept", False),
    "sit": ("sits", "sitting", "sat", False),
    "wait": ("waits", "waiting", "waited", False),
    "meet": ("meets", "meeting", "met", False),
    "visit": ("visits", "visiting", "visited", False),
}

VERB_INFL_TO_LEMMA = {}
for _lemma, (_v3, _vprog, _vpast, _) in VERBS.items():
    VERB_INFL_TO_LEMMA[_lemma] = _lemma
    VERB_INFL_TO_LEMMA[_v3] = _lemma
    VERB_INFL_TO_LEMMA[_vprog] = _lemma
    VERB_INFL_TO_LEMMA[_vpast] = _lemma

ISL_STOPWORDS = {
    "am", "is", "are", "was", "were", "be", "been", "being",
    "a", "an", "the", "to", "of", "at", "in", "on",
    "do", "does", "did", "will", "would", "shall",
    "please", "very", "really", "just",
}


def clean_input_text(raw_text: str) -> str:
    """
    Validate and clean user text / sign-gloss input:
    - Enforces type & length limits.
    - Normalizes Unicode, apostrophes, underscores, repeated punctuation, and whitespace.
    - Deduplicates adjacent repeated sign stutters from sliding-window capture.
    """
    if not isinstance(raw_text, str):
        raise ValueError("Sentence must be a text string.")

    text = unicodedata.normalize("NFKC", raw_text)
    if len(text) > MAX_PROMPT_LENGTH:
        raise ValueError(f"Input exceeds maximum length of {MAX_PROMPT_LENGTH} characters.")

    # Remove control characters while preserving printable spaces
    text = "".join(ch if ch.isprintable() or ch in "\t\n\r" else " " for ch in text)
    text = text.replace("’", "'").replace("‘", "'").replace("_", " ")
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text or not any(ch.isalnum() for ch in text):
        raise ValueError("Please provide a non-empty word or sentence to translate.")

    # Collapse consecutive multi-word sign repeats ("How are you How are you" -> "How are you")
    for phrase, canonical in MULTIWORD_PHRASES:
        pattern = re.compile(rf"\b({re.escape(phrase)})(?:\s+\1)+\b", re.IGNORECASE)
        text = pattern.sub(canonical, text)

    # Collapse immediate duplicate single-word stutters ("Hello Hello" -> "Hello")
    tokens = text.split(" ")
    deduped = []
    for tok in tokens:
        stripped = tok.strip(".,!?;:")
        prev_stripped = deduped[-1].strip(".,!?;:") if deduped else None
        if prev_stripped and stripped and stripped.lower() == prev_stripped.lower():
            # Preserve stronger trailing punctuation if present on repeated token
            if tok[-1:] in "?!." and deduped[-1][-1:] not in "?!.":
                deduped[-1] = deduped[-1] + tok[-1]
            continue
        deduped.append(tok)

    return " ".join(deduped).strip()


def _article_for(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _place_phrase(place_word: str, preposition: str = "to") -> str:
    lw = place_word.lower()
    if lw == "home":
        return "home" if preposition == "to" else "at home"
    if lw in {"school", "university", "college", "class"} and preposition == "to":
        return f"to {lw}"
    return f"{preposition} the {lw}"


def _copula_for(subj_key: str, tense: str = "present") -> str:
    if tense == "past":
        return "were" if subj_key in {"you", "we", "they", "plural"} else "was"
    if subj_key == "I":
        return "am"
    if subj_key in {"you", "we", "they", "plural"}:
        return "are"
    return "is"


def _have_for(subj_key: str) -> str:
    return "have" if subj_key in {"I", "you", "we", "they", "plural"} else "has"


def _do_for(subj_key: str, tense: str = "present") -> str:
    if tense == "past":
        return "did"
    return "do" if subj_key in {"I", "you", "we", "they", "plural"} else "does"


def _capitalize_sentence(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.;?!])", r"\1", s)
    s = re.sub(r"\bi\b", "I", s)
    if not s:
        return s
    return s[0].upper() + s[1:]


def _format_subject(subj_tokens: list[str]) -> tuple[str, str]:
    """Return (formatted_subject_phrase, agreement_key)."""
    low = [t.lower() for t in subj_tokens]
    if len(low) == 1:
        w = low[0]
        if w in PRONOUN_SUBJ:
            p = PRONOUN_SUBJ[w]
            return p, p
        if w in KINSHIP_PROPER:
            return KINSHIP_PROPER[w], "3sg"
        if w in PERSON_NOUNS or w in PROFESSIONS or w in OBJECTS_OWNABLE or w in PLACES:
            return f"the {w}", "3sg"
        return subj_tokens[0], "3sg"

    if len(low) == 2 and low[0] in PRONOUN_POSS and (low[1] in PERSON_NOUNS or low[1] in PROFESSIONS or low[1] in OBJECTS_OWNABLE or low[1] in PLACES):
        return f"{PRONOUN_POSS[low[0]]} {low[1]}", "3sg"

    if len(low) == 2 and low[0] in ADJECTIVES and (low[1] in OBJECTS_OWNABLE or low[1] in PERSON_NOUNS or low[1] in PLACES):
        return f"the {low[0]} {low[1]}", "3sg"

    if "and" in low:
        parts = [KINSHIP_PROPER.get(x, PRONOUN_SUBJ.get(x, x)) for x in low if x != "and"]
        return " and ".join(parts), "plural"

    # Two juxtaposed person nouns in ISL ("Father Mother happy" -> "Father and Mother")
    if len(low) == 2 and all(x in PERSON_NOUNS or x in PRONOUN_SUBJ for x in low):
        p0 = KINSHIP_PROPER.get(low[0], PRONOUN_SUBJ.get(low[0], f"the {low[0]}"))
        p1 = KINSHIP_PROPER.get(low[1], PRONOUN_SUBJ.get(low[1], f"the {low[1]}"))
        return f"{p0} and {p1}", "plural"

    return " ".join(subj_tokens), "3sg"


def _format_object_np(tokens: list[str]) -> str:
    low = [t.lower() for t in tokens]
    if not low:
        return ""
    if len(low) == 1:
        w = low[0]
        if w in {"water", "food", "money", "medicine", "help"}:
            return w
        if w in PRONOUN_SUBJ:
            obj_map = {"i": "me", "we": "us", "he": "him", "she": "her", "they": "them"}
            return obj_map.get(w, w)
        if w in KINSHIP_PROPER:
            return KINSHIP_PROPER[w]
        return f"{_article_for(w)} {w}"
    if len(low) == 2 and low[0] in PRONOUN_POSS:
        return f"{PRONOUN_POSS[low[0]]} {low[1]}"
    if len(low) == 2 and low[0] in ADJECTIVES:
        if low[1] in {"water", "food", "money", "medicine"}:
            return f"{low[0]} {low[1]}"
        return f"{_article_for(low[0])} {low[0]} {low[1]}"
    return " ".join(low)


def _is_already_natural_english(words: list[str], punct: str) -> bool:
    """
    Detect if a clause is already natural finite English rather than telegraphic ISL gloss.
    Carefully avoids false positives when copulas/articles are missing or wh/negation is clause-final.
    """
    low = [w.lower() for w in words]
    if not low:
        return True

    # Clause-final wh-word ("Hospital where", "you go School when") is ISL gloss order, NOT natural English
    if len(low) >= 2 and low[-1] in WH_WORDS:
        return False

    # Clause-final bare "not" without preceding auxiliary ("I Doctor not", "I School go not") is ISL gloss
    if len(low) >= 2 and low[-1] in {"not", "no"} and (len(low) < 3 or low[-2] not in AUX_AND_FINITE):
        return False

    has_aux = any(w in AUX_AND_FINITE for w in low)
    has_func = any(w in FUNCTION_MARKERS for w in low)

    if has_aux:
        return True

    # Check if there is an inflected/finite English verb in SVO position WITH function words (articles/prepositions)
    verb_positions = [i for i, w in enumerate(low) if w in VERB_INFL_TO_LEMMA]
    if verb_positions and has_func:
        # Still check missing "will" with leading "tomorrow" ("Tomorrow I go to school" -> needs "will")
        if low[0] == "tomorrow" and "will" not in low:
            return False
        return True

    return False


def _realize_single_clause(raw_clause: str) -> str:
    clause = raw_clause.strip()
    if not clause:
        return ""

    # Preserve terminal punctuation hint
    trailing_punct = ""
    if clause[-1] in "?!.":
        trailing_punct = clause[-1]
        clause_body = clause[:-1].strip()
    else:
        clause_body = clause

    # Replace canonical multi-word phrases with temporary atomic markers where helpful
    norm = clause_body
    norm = re.sub(r"\bgood\s+morning\b", "Good_Morning", norm, flags=re.IGNORECASE)
    norm = re.sub(r"\bgood\s+afternoon\b", "Good_Afternoon", norm, flags=re.IGNORECASE)
    norm = re.sub(r"\bgood\s+evening\b", "Good_Evening", norm, flags=re.IGNORECASE)
    norm = re.sub(r"\bgood\s+night\b", "Good_Night", norm, flags=re.IGNORECASE)
    norm = re.sub(r"\bthank\s+you\b", "Thank_You", norm, flags=re.IGNORECASE)
    norm = re.sub(r"\bhow\s+are\s+you\b", "How_Are_You", norm, flags=re.IGNORECASE)

    tokens = [t.strip(",") for t in norm.split() if t.strip(",")]
    if not tokens:
        return ""

    GREETING_MAP = {
        "hello": "Hello",
        "hi": "Hello",
        "hey": "Hello",
        "namaste": "Hello",
        "good_morning": "Good morning",
        "good_afternoon": "Good afternoon",
        "good_evening": "Good evening",
        "good_night": "Good night",
        "thank_you": "Thank you",
        "thanks": "Thank you",
    }

    # Peel leading greeting(s) cleanly ("Hello How_Are_You", "Good_Morning Mother", "Hello I Doctor")
    greetings = []
    idx = 0
    while idx < len(tokens):
        tl = tokens[idx].lower()
        if tl in GREETING_MAP:
            greetings.append(GREETING_MAP[tl])
            idx += 1
        elif tl == "how_are_you" and idx == 0 and len(tokens) == 1:
            return "How are you?"
        else:
            break

    rem = tokens[idx:]
    if greetings and not rem:
        joined = ", ".join(greetings)
        return _capitalize_sentence(f"{joined}!")

    if rem and rem[0].lower() == "how_are_you":
        tail = rem[1:]
        vocative = f", {KINSHIP_PROPER.get(tail[0].lower(), tail[0])}" if len(tail) == 1 else ""
        prefix = f"{', '.join(greetings)}, " if greetings else ""
        return _capitalize_sentence(f"{prefix}how are you{vocative}?")

    # Vocative right after greeting ("Good_Morning Mother", "Thank_You Doctor", "Hello Father")
    if greetings and len(rem) == 1 and (rem[0].lower() in PERSON_NOUNS or rem[0].lower() in PROFESSIONS):
        voc = KINSHIP_PROPER.get(rem[0].lower(), rem[0].capitalize())
        return _capitalize_sentence(f"{', '.join(greetings)}, {voc}!")

    # Expand remaining tokens back into plain words
    expanded = []
    for t in rem:
        if t.lower() == "how_are_you":
            expanded.extend(["how", "are", "you"])
        else:
            expanded.extend(t.replace("_", " ").split())

    prefix_greeting = f"{', '.join(greetings)}, " if greetings else ""

    # Single-word utterances
    if len(expanded) == 1:
        w = expanded[0]
        wl = w.lower()
        if wl == "pleased":
            return _capitalize_sentence(f"{prefix_greeting}I am pleased.")
        if wl in {"alright", "okay", "ok"}:
            return _capitalize_sentence(f"{prefix_greeting}Alright!")
        if wl in WH_WORDS:
            return _capitalize_sentence(f"{prefix_greeting}{wl}?")
        if wl == "no":
            return _capitalize_sentence(f"{prefix_greeting}No.")
        canon = VOCAB_LOWER_MAP.get(wl, w.capitalize())
        end_p = trailing_punct if trailing_punct else ""
        if prefix_greeting:
            return _capitalize_sentence(f"{prefix_greeting}{canon}{end_p or '.'}")
        return f"{canon}{end_p}" if canon == "I" else f"{canon[0].upper() + canon[1:]}{end_p}"

    # Already well-formed natural English sentence -> preserve natural phrasing with clean punctuation
    if _is_already_natural_english(expanded, trailing_punct):
        text_out = " ".join(expanded)
        is_q = trailing_punct == "?" or expanded[0].lower() in WH_WORDS or expanded[0].lower() in {"is", "are", "am", "do", "does", "did", "can", "could", "will", "would", "should", "have", "has"}
        end_p = "?" if is_q else (trailing_punct or ".")
        return _capitalize_sentence(f"{prefix_greeting}{text_out}{end_p}")

    # --- Compositional ISL Gloss / Telegraphic Clause Realization ---
    time_lead = ""
    tense = "present"
    negated = False
    wh_word = None
    is_question = trailing_punct == "?"

    content = []
    for tok in expanded:
        tl = tok.lower()
        if tl in TIME_WORDS and not time_lead:
            time_lead, tense = TIME_WORDS[tl]
        elif tl in NEGATORS:
            negated = True
        elif tl in WH_WORDS and wh_word is None:
            wh_word = tl
            is_question = True
        else:
            content.append(tok)

    time_prefix = f"{time_lead} " if time_lead else ""

    # Pure wh-question with single topic ("Hospital where" -> "Where is the hospital?", "Mother how" -> "How is Mother?")
    if wh_word and len(content) <= 2 and not any(t.lower() in VERB_INFL_TO_LEMMA for t in content):
        if not content:
            return _capitalize_sentence(f"{prefix_greeting}{wh_word}?")
        subj_str, subj_key = _format_subject(content)
        cop = _copula_for(subj_key, "past" if tense == "past" else "present")
        return _capitalize_sentence(f"{prefix_greeting}{wh_word} {cop} {subj_str}?")

    # Identify explicit verb (if any) in content tokens
    verb_idx = None
    verb_lemma = None
    for i, tok in enumerate(content):
        tl = tok.lower()
        if tl in VERB_INFL_TO_LEMMA:
            verb_idx = i
            verb_lemma = VERB_INFL_TO_LEMMA[tl]
            break

    # Case 1: Clause with an explicit action/motion verb (handles both SOV "I Book read" / "Tomorrow I School go" and telegraphic SVO "Tomorrow I go School")
    if verb_lemma is not None:
        other = content[:verb_idx] + content[verb_idx + 1:]
        # First subject-capable NP is Subject; remaining tokens form Place and/or Object
        if other:
            if len(other) >= 3 and other[0].lower() in PRONOUN_POSS and other[1].lower() in PERSON_NOUNS:
                subj_tokens = other[:2]
                rest = other[2:]
            else:
                subj_tokens = [other[0]]
                rest = other[1:]
        else:
            subj_tokens = ["you"]
            rest = []

        subj_str, subj_key = _format_subject(subj_tokens)
        places = [t for t in rest if t.lower() in PLACES]
        non_places = [t for t in rest if t.lower() not in PLACES]

        obj_phrase = _format_object_np(non_places) if non_places else ""
        v3, vprog, vpast, takes_to = VERBS[verb_lemma]
        prep = "to" if takes_to else "at"
        loc_phrase = _place_phrase(places[0], prep) if places else ""

        vp_tail = " ".join(p for p in [obj_phrase, loc_phrase] if p)
        tail_sp = f" {vp_tail}" if vp_tail else ""

        if is_question:
            q_lead = f"{wh_word} " if wh_word else ""
            if tense == "future":
                neg_str = "not " if negated else ""
                body = f"{q_lead}will {subj_str} {neg_str}{verb_lemma}{tail_sp}"
            elif tense == "past":
                neg_str = "not " if negated else ""
                body = f"{q_lead}did {subj_str} {neg_str}{verb_lemma}{tail_sp}"
            elif tense == "present_prog":
                cop = _copula_for(subj_key, "present")
                neg_str = "not " if negated else ""
                body = f"{q_lead}{cop} {subj_str} {neg_str}{vprog}{tail_sp}"
            else:
                aux = _do_for(subj_key, "present")
                neg_str = "not " if negated else ""
                body = f"{q_lead}{aux} {subj_str} {neg_str}{verb_lemma}{tail_sp}"
            return _capitalize_sentence(f"{prefix_greeting}{time_prefix}{body}?")

        # Declarative verbal clause
        if tense == "future":
            vp = f"will not {verb_lemma}" if negated else f"will {verb_lemma}"
        elif tense == "past":
            vp = f"did not {verb_lemma}" if negated else vpast
        elif tense == "present_prog":
            cop = _copula_for(subj_key, "present")
            vp = f"{cop} not {vprog}" if negated else f"{cop} {vprog}"
        else:
            if negated:
                vp = f"{_do_for(subj_key, 'present')} not {verb_lemma}"
            else:
                vp = verb_lemma if subj_key in {"I", "you", "we", "they", "plural"} else v3

        return _capitalize_sentence(f"{prefix_greeting}{time_prefix}{subj_str} {vp}{tail_sp}.")

    # Case 2: Verbless clauses (Copular, Motion-to-Place, Possessive)
    # Split content into Subject NP + Predicate tokens
    if len(content) >= 3 and content[0].lower() in PRONOUN_POSS and content[1].lower() in PERSON_NOUNS:
        subj_tokens = content[:2]
        pred_tokens = content[2:]
    elif len(content) >= 3 and all(x.lower() in PERSON_NOUNS for x in content[:2]):
        subj_tokens = content[:2]
        pred_tokens = content[2:]
    elif len(content) >= 3 and content[0].lower() in ADJECTIVES and content[1].lower() in (OBJECTS_OWNABLE | PERSON_NOUNS | PLACES):
        subj_tokens = content[:2]
        pred_tokens = content[2:]
    else:
        subj_tokens = content[:1]
        pred_tokens = content[1:]

    subj_str, subj_key = _format_subject(subj_tokens)
    pred_low = [t.lower() for t in pred_tokens]

    # 2A: Predicate is a Place ("Tomorrow I School", "Today I Hospital", "I Hospital not", "Mother Market")
    if pred_low and pred_low[-1] in PLACES:
        dest = _place_phrase(pred_low[-1], "to")
        if is_question:
            q_lead = f"{wh_word} " if wh_word else ""
            if tense == "future":
                return _capitalize_sentence(f"{prefix_greeting}{time_prefix}{q_lead}will {subj_str} {'not ' if negated else ''}go {dest}?")
            cop = _copula_for(subj_key, "past" if tense == "past" else "present")
            return _capitalize_sentence(f"{prefix_greeting}{time_prefix}{q_lead}{cop} {subj_str} {'not ' if negated else ''}going {dest}?")
        if tense == "future":
            vp = "will not go" if negated else "will go"
            return _capitalize_sentence(f"{prefix_greeting}{time_prefix}{subj_str} {vp} {dest}.")
        if tense == "past":
            vp = "did not go" if negated else "went"
            return _capitalize_sentence(f"{prefix_greeting}{time_prefix}{subj_str} {vp} {dest}.")
        cop = _copula_for(subj_key, "present")
        vp = f"{cop} not going" if negated else f"{cop} going"
        return _capitalize_sentence(f"{prefix_greeting}{time_prefix}{subj_str} {vp} {dest}.")

    # 2B: Predicate is Adjective ("I happy", "Father Mother happy", "Dog White", "I cold", "I happy not")
    if pred_low and all(p in ADJECTIVES for p in pred_low):
        adj_str = " ".join(pred_low)
        cop = _copula_for(subj_key, "past" if tense == "past" else "present")
        if is_question:
            q_lead = f"{wh_word} " if wh_word else ""
            return _capitalize_sentence(f"{prefix_greeting}{time_prefix}{q_lead}{cop} {subj_str} {'not ' if negated else ''}{adj_str}?")
        vp = f"{cop} not {adj_str}" if negated else f"{cop} {adj_str}"
        return _capitalize_sentence(f"{prefix_greeting}{time_prefix}{subj_str} {vp}.")

    # 2C: Predicate is Profession / Role ("I Doctor", "Mother Teacher", "you Lawyer ?")
    if pred_low and pred_low[-1] in PROFESSIONS:
        role_np = _format_object_np(pred_tokens) if subj_key != "plural" else f"{pred_low[-1]}s"
        cop = _copula_for(subj_key, "past" if tense == "past" else "present")
        if is_question:
            q_lead = f"{wh_word} " if wh_word else ""
            return _capitalize_sentence(f"{prefix_greeting}{time_prefix}{q_lead}{cop} {subj_str} {'not ' if negated else ''}{role_np}?")
        vp = f"{cop} not {role_np}" if negated else f"{cop} {role_np}"
        return _capitalize_sentence(f"{prefix_greeting}{time_prefix}{subj_str} {vp}.")

    # 2D: Animate Subject + Ownable Concrete Object ("I Book", "I Red Book", "Father Black Car", "you Pen")
    if pred_low and pred_low[-1] in OBJECTS_OWNABLE and (subj_tokens[0].lower() in PRONOUN_SUBJ or subj_tokens[0].lower() in PERSON_NOUNS):
        obj_np = _format_object_np(pred_tokens)
        if is_question:
            q_lead = f"{wh_word} " if wh_word else ""
            aux = _do_for(subj_key, tense)
            return _capitalize_sentence(f"{prefix_greeting}{time_prefix}{q_lead}{aux} {subj_str} {'not ' if negated else ''}have {obj_np}?")
        if negated:
            return _capitalize_sentence(f"{prefix_greeting}{time_prefix}{subj_str} {_do_for(subj_key, tense)} not have {obj_np}.")
        return _capitalize_sentence(f"{prefix_greeting}{time_prefix}{subj_str} {_have_for(subj_key)} {obj_np}.")

    # 2E: General fallback nominal predicate ("X is Y")
    cop = _copula_for(subj_key, "past" if tense == "past" else "present")
    rest_str = " ".join(pred_tokens)
    if is_question:
        q_lead = f"{wh_word} " if wh_word else ""
        return _capitalize_sentence(f"{prefix_greeting}{time_prefix}{q_lead}{cop} {subj_str} {'not ' if negated else ''}{rest_str}?".strip())
    if rest_str:
        vp = f"{cop} not {rest_str}" if negated else f"{cop} {rest_str}"
        return _capitalize_sentence(f"{prefix_greeting}{time_prefix}{subj_str} {vp}.")
    vp = f"{cop} not" if negated else ""
    return _capitalize_sentence(f"{prefix_greeting}{time_prefix}{subj_str} {vp}.".replace(" .", "."))


def reconstruct_english_sentence(raw_text: str) -> str:
    """
    Clean input text and reconstruct natural, grammatically well-formed English
    from either raw ISL gloss sequences (SOV / copula-drop / article-drop / wh-final / neg-final)
    or user text prompts.
    """
    cleaned = clean_input_text(raw_text)
    # Split on sentence-ending punctuation while preserving delimiters
    parts = re.findall(r"[^.!?]+[.!?]?", cleaned)
    realized = []
    for p in parts:
        p = p.strip()
        if not p or not any(c.isalnum() for c in p):
            continue
        out = _realize_single_clause(p)
        if out:
            realized.append(out)
    return " ".join(realized) if realized else cleaned


def _lemma_english_token(word: str) -> str:
    lw = word.lower().strip(".,!?;:'\"")
    if lw in SYNONYM_TO_SIGN:
        return SYNONYM_TO_SIGN[lw]
    if lw in VERB_INFL_TO_LEMMA:
        v = VERB_INFL_TO_LEMMA[lw]
        return SYNONYM_TO_SIGN.get(v, v)
    # Simple regular plural stripping ("books" -> "Book", "doctors" -> "Doctor", "libraries" -> "Library")
    if lw.endswith("ies") and len(lw) > 4:
        cand = lw[:-3] + "y"
        if cand in SYNONYM_TO_SIGN:
            return SYNONYM_TO_SIGN[cand]
    if lw.endswith("es") and len(lw) > 3:
        cand = lw[:-2]
        if cand in SYNONYM_TO_SIGN:
            return SYNONYM_TO_SIGN[cand]
    if lw.endswith("s") and len(lw) > 2:
        cand = lw[:-1]
        if cand in SYNONYM_TO_SIGN:
            return SYNONYM_TO_SIGN[cand]
    return VOCAB_LOWER_MAP.get(lw, word)


def text_to_isl_gloss(text: str) -> dict:
    """
    Convert an English sentence or gloss sequence into canonical ISL SOV Gloss order
    (Time -> Subject -> Place/Object -> Verb -> Negation -> Wh-Question)
    and attach matching sign metadata & right-hand gesture GIF paths from the 70-class vocabulary.
    """
    cleaned = clean_input_text(text)
    work = cleaned
    phrase_tokens = []
    # Match multi-word ISL signs cleanly first
    for phrase, canon in MULTIWORD_PHRASES:
        pat = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
        work = pat.sub(canon.replace(" ", "__"), work)

    raw_tokens = [t.strip(".,!?;:\"'()[]") for t in work.split()]
    greetings, times, subjects, places_objs, verbs, negs, whs = [], [], [], [], [], [], []

    for tok in raw_tokens:
        if not tok:
            continue
        unpacked = tok.replace("__", " ")
        ul = unpacked.lower()

        if ul in {"don't", "dont", "do not", "doesn't", "didn't", "won't", "can't", "cannot", "isn't", "aren't", "wasn't", "weren't"}:
            negs.append("NOT")
            continue
        if ul in NEGATORS:
            negs.append("NOT" if ul != "never" else "NEVER")
            continue
        if ul in WH_WORDS:
            whs.append(ul.upper())
            continue
        if ul in ISL_STOPWORDS:
            continue

        mapped = VOCAB_LOWER_MAP.get(ul) or _lemma_english_token(unpacked)
        ml = mapped.lower()

        if mapped in {"Hello", "Good Morning", "Good night", "Thank you", "How are you"}:
            greetings.append(mapped)
        elif ml in TIME_WORDS:
            times.append(mapped)
        elif ml in VERB_INFL_TO_LEMMA and mapped not in VOCAB_70:
            verbs.append(VERB_INFL_TO_LEMMA[ml].upper())
        elif not subjects and (ml in PRONOUN_SUBJ or ml in PERSON_NOUNS):
            subjects.append(mapped)
        else:
            places_objs.append(mapped)

    ordered = greetings + times + subjects + places_objs + verbs + negs + whs
    gloss_tokens = [t.upper() for t in ordered]

    sign_sequence = []
    seen_consecutive = None
    for item in ordered:
        canon = VOCAB_LOWER_MAP.get(item.lower())
        if canon:
            if canon == seen_consecutive:
                continue
            seen_consecutive = canon
            gif_name = f"{canon.replace(' ', '_')}.gif"
            sign_sequence.append({
                "word": canon,
                "gloss": canon.upper(),
                "matched": True,
                "gif_url": f"/static/gesture_gifs/{gif_name}",
                "hint": GESTURE_HINTS.get(canon, ""),
            })
        else:
            seen_consecutive = None
            sign_sequence.append({
                "word": item,
                "gloss": item.upper(),
                "matched": False,
                "gif_url": None,
                "hint": "Fingerspelled / lexical gloss concept",
            })

    return {
        "isl_gloss": " ".join(gloss_tokens),
        "isl_tokens": gloss_tokens,
        "sign_sequence": sign_sequence,
    }
