"""
Build NLP correction lexicon from approved labels.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_quality import load_csv_dataframe
from src.utils.lexicon import save_word_counts, tokenize_words
from src.config import LABEL_CSV, LEXICON_PATH


def build_word_counts(csv_path: Path) -> dict:
    from collections import Counter
    df = load_csv_dataframe(csv_path)
    
    # Optional: only use approved if status exists
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.strip().str.lower() == "approved"]

    all_text = " ".join(df["text"].astype(str).tolist())
    tokens = [t.lower() for t in tokenize_words(all_text)]
    return Counter(tokens)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(LABEL_CSV), help="CSV source.")
    parser.add_argument("--output", default=str(LEXICON_PATH), help="JSON output.")
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)

    counts = build_word_counts(source)
    save_word_counts(counts, output)

    print(f"Lexicon words: {len(counts)}")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
