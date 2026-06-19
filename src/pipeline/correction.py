"""
NLP text correction pipeline.
Combines IndoBERT deep learning correction with lexicon-based correction.
"""

from __future__ import annotations

import logging
from collections import Counter

from src.inference.indobert import IndoBERTCorrector
from src.utils.lexicon import correct_text

logger = logging.getLogger("notescanin.correction")


def nlp_correct(
    text: str,
    indobert: IndoBERTCorrector | None = None,
    lexicon: Counter | None = None,
) -> str:
    """
    Apply NLP post-processing pipeline:
        1. IndoBERT typo correction (deep learning)
        2. Lexicon-based correction (rule-based)
    """
    corrected = text

    # Step 1: IndoBERT
    if indobert is not None and indobert.is_loaded:
        corrected = indobert.correct(corrected)

    # Step 2: Lexicon
    if lexicon is not None:
        corrected = correct_text(corrected, lexicon)

    return corrected
