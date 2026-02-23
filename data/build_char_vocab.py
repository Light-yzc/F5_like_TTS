"""
Build character vocabulary from dataset content.txt and save as JSON.

Usage:
    python data/build_char_vocab.py --data_root data/processed/ --output data/char_vocab.json

Reads content.txt (format: "speaker_uttId_text" per line),
extracts all unique characters, and saves the vocabulary.
"""

import os
import json
import argparse
from collections import Counter, defaultdict
from utils.g2p_ipa import g2p_ipa_batch

BATCH_SIZE = 500  # Process 500 texts per espeak-ng call


def build_vocab(data_root: str) -> dict[str, int]:
    """Build character vocab from content.txt using batch G2P."""
    content_path = os.path.join(data_root, "content.txt")
    if not os.path.exists(content_path):
        raise FileNotFoundError(f"content.txt not found at {content_path}")

    # ── Pass 1: Group texts by language ──
    lang_texts: dict[str, list[str]] = defaultdict(list)

    with open(content_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("_", 2)
            if len(parts) < 3:
                continue
            speaker = parts[0]
            text = parts[2]
            if speaker.startswith("jvs"):
                language = "JA"
            elif speaker.startswith("SSB"):
                language = "ZH"
            elif parts[1].startswith("char"):
                language = "JA"
            else:
                language = "EN"
            lang_texts[language].append(text)

    # ── Pass 2: Batch phonemize per language ──
    char_counter = Counter()
    num_lines = 0

    for lang, texts in lang_texts.items():
        print(f"  Phonemizing {len(texts)} {lang} texts in batches of {BATCH_SIZE}...")
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            ipa_results = g2p_ipa_batch(batch, lang)
            for ipa in ipa_results:
                tagged = f"[{lang}] {ipa}"
                char_counter.update(tagged)
            num_lines += len(batch)

    # Ensure structural tokens used in dataset/inference are in vocab
    char_counter.update("[SEP]")

    # Build vocab: PAD=0, UNK=1, then sorted by frequency (most frequent first)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for char, _count in char_counter.most_common():
        vocab[char] = len(vocab)

    print(f"Scanned {num_lines} lines")
    print(f"Unique characters: {len(vocab) - 2}")
    print(f"Total vocab size: {len(vocab)}")
    print(f"Top 20 chars: {[ch for ch, _ in char_counter.most_common(20)]}")

    return vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to processed data directory (containing content.txt)")
    parser.add_argument("--output", type=str, default="data/char_vocab.json",
                        help="Output path for vocab JSON")
    args = parser.parse_args()

    vocab = build_vocab(args.data_root)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"\nVocab saved to {args.output}")


if __name__ == "__main__":
    main()
