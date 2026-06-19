"""
Hybrid fusion of TrOCR and CRNN predictions.
Combines outputs using lexicon-based scoring.
"""

from __future__ import annotations

import logging
from collections import Counter

from src.utils.lexicon import correct_text, score_text, tokenize_words

logger = logging.getLogger("notescanin.fusion")


def fuse_predictions(
    crnn_text: str,
    trocr_text: str,
    lexicon: Counter,
    crnn_confidence: float | None = None,
) -> str:
    """
    Fuse CRNN and TrOCR predictions using lexicon-based scoring.

    Strategy:
        1. If either is empty, use the other
        2. If both correct to the same text, use that
        3. If CRNN confidence is very low, prefer TrOCR
        4. Score both against lexicon and pick the best
        5. Attempt token-level fusion for same-length predictions
    """
    crnn_text = str(crnn_text).strip()
    trocr_text = str(trocr_text).strip()

    if not crnn_text:
        return correct_text(trocr_text, lexicon)
    if not trocr_text:
        return correct_text(crnn_text, lexicon)

    corrected_crnn = correct_text(crnn_text, lexicon)
    corrected_trocr = correct_text(trocr_text, lexicon)

    if corrected_crnn == corrected_trocr:
        return corrected_crnn

    if crnn_confidence is not None and crnn_confidence < 0.08:
        return corrected_trocr

    score_crnn = score_text(corrected_crnn, lexicon)
    score_trocr = score_text(corrected_trocr, lexicon)

    crnn_tokens = tokenize_words(corrected_crnn)
    trocr_tokens = tokenize_words(corrected_trocr)

    if score_trocr >= score_crnn - 0.5:
        return corrected_trocr

    if (
        crnn_tokens
        and trocr_tokens
        and len(crnn_tokens) == len(trocr_tokens)
        and abs(score_crnn - score_trocr) <= 1.0
    ):
        fused: list[str] = []
        for ct, tt in zip(crnn_tokens, trocr_tokens):
            crnn_known = ct.lower() in lexicon
            trocr_known = tt.lower() in lexicon

            if ct.lower() == tt.lower():
                fused.append(ct)
            elif crnn_known and not trocr_known:
                fused.append(ct)
            elif trocr_known and not crnn_known:
                fused.append(tt)
            elif len(ct) >= len(tt):
                fused.append(ct)
            else:
                fused.append(tt)

        return correct_text(" ".join(fused), lexicon)

    if (
        crnn_confidence is not None
        and crnn_confidence > 0.35
        and score_crnn > score_trocr + 1.5
    ):
        return corrected_crnn

    return corrected_trocr if score_trocr >= score_crnn else corrected_crnn
