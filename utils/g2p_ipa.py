import os
import logging

# Fix Colab /tmp noexec: point to system espeak-ng directly
if not os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
    for lib_path in [
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",  # Ubuntu/Colab
        "/usr/lib/libespeak-ng.so.1",                     # Other Linux
        "/opt/homebrew/lib/libespeak-ng.dylib",            # macOS ARM
        "/usr/local/lib/libespeak-ng.dylib",               # macOS Intel
    ]:
        if os.path.exists(lib_path):
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib_path
            break

# Suppress "words count mismatch" warning spam from phonemizer
logging.getLogger("phonemizer").setLevel(logging.ERROR)

"""
IPA-based G2P for multilingual TTS.

Uses `phonemizer` (espeak-ng backend) to convert ZH/JA/EN text to a unified
International Phonetic Alphabet representation. This means shared sounds across
languages map to the SAME tokens — e.g., /t/ in Chinese, Japanese, and English
are identical, dramatically reducing the text encoder's learning burden.

Dependencies:
    pip install phonemizer
    # Plus system-level: brew install espeak-ng  (macOS)
    #                     apt install espeak-ng   (Ubuntu)

Comparison with current approach (pinyin/romaji/lowercase):
    Current:  ZH "今天" → "jin1 tian1"   (pinyin chars, tone numbers)
              JA "今日" → "kyou"          (romaji chars)
              EN "today" → "t o d a y"    (lowercase chars)
              → Same letter 't' means different things in each language!

    IPA:      ZH "今天" → "tɕintʰjɛn"    (IPA phonemes)
              JA "今日" → "kʲoː"          (IPA phonemes)
              EN "today" → "tʊdeɪ"        (IPA phonemes)
              → Shared sounds like /t/ are identical tokens across languages!
"""

import re
from functools import lru_cache

try:
    from phonemizer import phonemize
    from phonemizer.separator import Separator
    from phonemizer.backend import EspeakBackend
    HAS_PHONEMIZER = True
except ImportError:
    HAS_PHONEMIZER = False


# ─── Language code mapping ───────────────────────────────────────────
# phonemizer/espeak-ng uses its own language codes
LANG_MAP = {
    "ZH": "cmn",       # Mandarin Chinese
    "JA": "ja",         # Japanese
    "EN": "en-us",      # American English
}


# ─── IPA Phoneme Separator ──────────────────────────────────────────
# We use space between words, nothing between phones within a word.
# This gives us a compact but parseable output.
IPA_SEP = Separator(phone="", word=" ", syllable="")


# ─── Core functions ─────────────────────────────────────────────────

def g2p_ipa(text: str, language: str) -> str:
    """
    Convert text to IPA using espeak-ng via phonemizer.

    Args:
        text:     input text in any supported language
        language: "ZH", "JA", or "EN"

    Returns:
        IPA string, e.g. "tɕintʰjɛn tʰjɛntɕʰi tʂənxau"
    """
    if not HAS_PHONEMIZER:
        raise ImportError(
            "Please install phonemizer: pip install phonemizer\n"
            "And espeak-ng: brew install espeak-ng (macOS) / "
            "apt install espeak-ng (Linux)"
        )

    lang_code = LANG_MAP.get(language.upper())
    if lang_code is None:
        raise ValueError(f"Unsupported language: {language}. Supported: {list(LANG_MAP.keys())}")

    # phonemize expects a list of strings
    result = phonemize(
        [text],
        language=lang_code,
        backend="espeak",
        separator=IPA_SEP,
        strip=True,
        preserve_punctuation=True,
        with_stress=False,        # Remove stress marks for simplicity
    )

    return result[0] if result else ""


def text_to_phonemes_ipa(text: str, language: str) -> str:
    """
    Drop-in replacement for text_to_phonemes() in g2p.py.
    Converts text to IPA and prepends a language tag.

    Example:
        text_to_phonemes_ipa("今天天气真好", "ZH")
        → "[ZH] tɕintʰjɛn tʰjɛntɕʰi tʂənxau"
    """
    language = language.upper()
    ipa = g2p_ipa(text, language)
    return f"[{language}] {ipa}"


def build_ipa_vocab(texts_with_langs: list[tuple[str, str]]) -> dict[str, int]:
    """
    Build a character-level vocab from IPA outputs.
    Since IPA uses a small set of unicode phoneme symbols (~100-150),
    the resulting vocab is compact and shared across all languages.

    Args:
        texts_with_langs: list of (text, language) tuples

    Returns:
        vocab dict: {"<PAD>": 0, "<UNK>": 1, "t": 2, "ɕ": 3, ...}
    """
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text, lang in texts_with_langs:
        ipa = text_to_phonemes_ipa(text, lang)
        for ch in ipa:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


# ─── Demo & Comparison ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("IPA G2P Demo — Unified phoneme space across languages")
    print("=" * 60)

    test_cases = [
        ("今天天气真好，我们去玩吧！", "ZH"),
        ("我的名字叫Jack。", "ZH"),
        ("サポートシステム、40%カット……運動性、問題ありません！", "JA"),
        ("Hello world, this is a test.", "EN"),
        ("I'm an apple's! How are you?", "EN"),
    ]

    print("\n--- IPA Output ---")
    all_chars = set()
    for text, lang in test_cases:
        ipa = text_to_phonemes_ipa(text, lang)
        print(f"  {lang}: {ipa}")
        all_chars.update(ipa)

    print(f"\n--- Vocab Stats ---")
    print(f"  Total unique IPA characters: {len(all_chars)}")
    print(f"  Characters: {''.join(sorted(all_chars))}")

    # Compare with current approach
    print("\n--- Comparison with current G2P ---")
    try:
        from utils.g2p import text_to_phonemes as old_g2p
        for text, lang in test_cases[:3]:
            old = old_g2p(text, lang)
            new = text_to_phonemes_ipa(text, lang)
            print(f"  {lang} OLD: {old}")
            print(f"  {lang} NEW: {new}")
            print()
    except ImportError:
        print("  (old g2p not available for comparison)")
