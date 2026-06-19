"""
Lexicon-based text correction utilities.
Uses word frequency data and edit distance for fuzzy matching.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from difflib import get_close_matches
from pathlib import Path
from typing import Iterable

WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9']+")
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9']+|[^A-Za-zÀ-ÿ0-9']+")

COMMON_ABBREVIATIONS = {
    "yg": "yang", "dg": "dengan", "pd": "pada", "dr": "dari",
    "utk": "untuk", "tdk": "tidak", "dll": "dan lain-lain",
    "dlm": "dalam", "sbg": "sebagai", "bs": "bisa", "krn": "karena",
    "sm": "sama", "tp": "tapi", "tpn": "tanpa", "kpd": "kepada",
    "sdh": "sudah", "udh": "udah", "ygs": "yang",
}


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(str(text))


def load_word_counts(path: Path | str) -> Counter:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Counter(payload)


def save_word_counts(counts: Counter, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def edit_distance(source: str, target: str) -> int:
    previous = list(range(len(target) + 1))
    for i, sc in enumerate(source, start=1):
        current = [i]
        for j, tc in enumerate(target, start=1):
            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (sc != tc),
            ))
        previous = current
    return previous[-1]


def normalized_similarity(source: str, target: str) -> float:
    max_len = max(len(source), len(target), 1)
    return 1.0 - (edit_distance(source, target) / max_len)


def _min_similarity(word: str) -> float:
    length = len(word)
    if length <= 3:
        return 0.80
    if length <= 5:
        return 0.74
    return 0.70


def _match_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


def choose_candidate(word: str, lexicon: Counter) -> str | None:
    lower = word.lower()

    if lower in COMMON_ABBREVIATIONS:
        return COMMON_ABBREVIATIONS[lower]
    if lower in lexicon:
        return lower
    if len(lower) < 3 or lower.isdigit():
        return None

    min_sim = _min_similarity(lower)
    candidates = get_close_matches(
        lower, list(lexicon.keys()), n=8,
        cutoff=max(0.60, min_sim - 0.08),
    )
    if not candidates:
        return None

    scored: list[tuple[float, float, int, str]] = []
    for cand in candidates:
        sim = normalized_similarity(lower, cand)
        if sim < min_sim:
            continue
        freq = lexicon[cand]
        dist = abs(len(cand) - len(lower))
        bonus = min(math.log1p(freq), 5.0) * 0.03
        scored.append((sim + bonus, sim, -dist, cand))

    return max(scored)[3] if scored else None


def correct_text(text: str, lexicon: Counter) -> str:
    """Correct text using lexicon-based fuzzy matching."""
    parts = TOKEN_RE.findall(str(text))
    output: list[str] = []

    for part in parts:
        if not WORD_RE.fullmatch(part):
            output.append(part)
            continue
        candidate = choose_candidate(part, lexicon)
        output.append(_match_case(part, candidate) if candidate else part)

    corrected = "".join(output)
    return re.sub(r"\s+", " ", corrected).strip()


def score_text(text: str, lexicon: Counter) -> float:
    """Score text against lexicon — higher is better."""
    tokens = tokenize_words(text)
    if not tokens:
        return float("-inf")

    known = 0.0
    penalty = 0.0
    prev_lower = None
    repeat_count = 0
    for token in tokens:
        lower = token.lower()
        if lower == prev_lower:
            repeat_count += 1
            penalty += 2.0 * repeat_count
            continue
            
        repeat_count = 0
        if lower in COMMON_ABBREVIATIONS:
            lower = COMMON_ABBREVIATIONS[lower]
        if lower in lexicon:
            known += 1.0 + math.log1p(lexicon[lower])
        else:
            penalty += 0.5
            
        prev_lower = lower

    return known - penalty
