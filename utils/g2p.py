import re
try:
    import pypinyin
    from pypinyin import Style, lazy_pinyin
    import rjieba
except ImportError:
    pypinyin = None
    rjieba = None

try:
    import pykakasi
    kakasi = pykakasi.kakasi()
except ImportError:
    pykakasi = None
    kakasi = None

def is_chinese_char(c):
    return '\u4e00' <= c <= '\u9fff'

def g2p_zh(text: str) -> str:
    """
    Convert Chinese text to Pinyin (with tones), keeping punctuation and English chars intact.
    Mimics F5-TTS's approach using rjieba for segmentation to handle polyphones.
    """
    if pypinyin is None or rjieba is None:
        raise ImportError("Please install pypinyin and rjieba for Chinese G2P.")
    
    char_list = []
    # Basic normalization
    text = text.replace(";", ",").replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    for seg in rjieba.cut(text):
        seg_byte_len = len(bytes(seg, "UTF-8"))
        if seg_byte_len == len(seg):  # Pure alphabets/symbols (English, numbers, etc.)
            if char_list and len(seg) > 1 and char_list[-1] not in " :'\"":
                char_list.append(" ")
            char_list.append(seg)
        elif seg_byte_len == 3 * len(seg):  # Pure Chinese characters
            pinyin_list = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
            for i, c in enumerate(seg):
                if is_chinese_char(c) and char_list and char_list[-1] != " ":
                    char_list.append(" ")
                char_list.append(pinyin_list[i])
        else:  # Mixed (e.g., "A股")
            for c in seg:
                if ord(c) < 256: # ASCII
                    char_list.append(c)
                elif is_chinese_char(c):
                    if char_list and char_list[-1] != " ":
                        char_list.append(" ")
                    char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                else:
                    char_list.append(c)
    
    return "".join(char_list).strip()

def g2p_ja(text: str) -> str:
    """
    Convert Japanese text (Kanji, Hiragana, Katakana) to Romaji.
    """
    if kakasi is None:
        raise ImportError("Please install pykakasi for Japanese G2P.")
    
    # pykakasi directly converts mixed text to romaji, preserving spaces/punctuation 
    # if it doesn't recognize them.
    result = kakasi.convert(text)
    romaji_parts = []
    for item in result:
        romaji_parts.append(item['hepburn']) # We use hepburn romanization
    
    # Join parts, optionally add spaces if needed for clarity between words, 
    # but Kakasi usually segments reasonably. We'll join with a space to make it look like words.
    # Joining without spaces might make the words too long, so space separation is safer for the tokenizer.
    return " ".join(romaji_parts).strip()

def g2p_en(text: str) -> str:
    """
    English processing. 
    Instead of full character-wise splitting, we separate words and punctuation,
    making it word-level but preserving punctuation marks.
    Example: "I'm an apple's!" -> "I ' m an apple ' s !"
    """
    # Use regex to find alphanumeric words or individual punctuation marks
    text = text.lower()
    # Split into individual characters to match the CharTokenizer approach
    # We still keep spaces to denote token barriers, but each letter is a token
    tokens = list(text)
    
    # Re-join with spaces to ensure tokenizer treats them as distinct units
    return " ".join(tokens).strip()

def text_to_phonemes(text: str, language: str) -> str:
    """
    Route the text to the appropriate G2P engine based on language,
    and prepend a language tag to explicitly condition the text encoder.
    """
    language = language.upper()
    if language == "ZH":
        phonemes = g2p_zh(text)
    elif language == "JA":
        phonemes = g2p_ja(text)
    elif language == "EN":
        phonemes = g2p_en(text)
    else:
        # Fallback to English basically (raw characters)
        phonemes = text
        
    return f"[{language}] {phonemes}"

if __name__ == "__main__":
    # Test cases
    print("ZH:", text_to_phonemes("今天天气真好，我们去玩吧！", "ZH"))
    print("ZH (Mixed):", text_to_phonemes("我的名字叫Jack。", "ZH"))
    print("JA:", text_to_phonemes("サポートシステム、40%カット……運動性、問題ありません！", "JA"))
    print("EN:", text_to_phonemes("Hello world, this is a test.", "EN"))
    print("EN (Punctuation):", text_to_phonemes("I'm an apple's! How are you?", "EN"))
