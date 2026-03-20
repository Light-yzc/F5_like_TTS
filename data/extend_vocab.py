"""
Extend existing character vocabulary with new characters from additional datasets.

Existing character IDs are preserved. New characters are appended at the end.

Usage:
    python extend_vocab.py --existing data/char_vocab.json --new_data_root new_data/ --output data/char_vocab.json
"""

import os
import json
import argparse
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.g2p_ipa import text_to_phonemes_ipa as text_to_phonemes


EROGE_TAG = "[EROGE]"


def extend_vocab(existing_path: str, new_data_root: str) -> dict[str, int]:
    """Load existing vocab and append new characters from new dataset."""
    # Load existing vocab
    with open(existing_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    old_size = len(vocab)

    # Scan new dataset for new characters
    content_path = os.path.join(new_data_root, "content.txt")
    if not os.path.exists(content_path):
        raise FileNotFoundError(f"content.txt not found at {content_path}")

    new_chars = []
    with open(content_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("_", 2)
            if len(parts) < 3:
                continue
            speaker = parts[0]
            utt_id = parts[1]
            if speaker == "none":
                language = "JA"
                if EROGE_TAG not in vocab:
                    vocab[EROGE_TAG] = len(vocab)
                    new_chars.append(EROGE_TAG)
            elif speaker.startswith("jvs"):
                language = "JA"
            elif speaker.startswith("SSB"):
                language = "ZH"
            elif utt_id.startswith("char"):
                language = "JA"
            else:
                language = "EN"
            text = parts[2]
            phonemes = text_to_phonemes(text, language)

            for ch in phonemes:
                if ch not in vocab:
                    vocab[ch] = len(vocab)
                    new_chars.append(ch)

    print(f"Existing vocab: {old_size} chars")
    print(f"New characters found: {len(new_chars)}")
    print(f"Updated vocab: {len(vocab)} chars")
    if new_chars:
        print(f"New chars: {new_chars[:50]}{'...' if len(new_chars) > 50 else ''}")

    return vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", type=str, required=True,
                        help="Path to existing char_vocab.json")
    parser.add_argument("--new_data_root", type=str, required=True,
                        help="Path to new dataset directory (containing content.txt)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: overwrite existing)")
    args = parser.parse_args()

    output = args.output or args.existing
    vocab = extend_vocab(args.existing, args.new_data_root)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
