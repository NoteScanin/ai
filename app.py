"""
NoteScanin AI Service — Entry Point
====================================
Run with: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI

from src.api import routes
from src.config import CRNN_CHECKPOINT, CRNN_VOCAB, INDOBERT_MODEL_DIR, LEXICON_PATH, TROCR_MODEL_DIR
from src.inference.crnn import CRNNInference
from src.inference.indobert import IndoBERTCorrector
from src.inference.trocr import TrOCRInference
from src.utils.lexicon import load_word_counts

logger = logging.getLogger("notescanin")


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load all models at startup, cleanup at shutdown."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting NoteScanin AI on %s", device)

    # TrOCR (menggunakan model HF generic sebagai fallback sementara)
    routes.trocr = TrOCRInference(model_dir=TROCR_MODEL_DIR, device=device)
    routes.trocr.load()

    # CRNN
    routes.crnn = CRNNInference(checkpoint=CRNN_CHECKPOINT, vocab=CRNN_VOCAB)
    routes.crnn.load()

    # IndoBERT
    routes.indobert = IndoBERTCorrector(model_dir=INDOBERT_MODEL_DIR, device=device)
    routes.indobert.load()

    # Lexicon
    lexicon_path = Path(LEXICON_PATH)
    if lexicon_path.exists():
        routes.lexicon = load_word_counts(lexicon_path)
        logger.info("Lexicon loaded: %d words", len(routes.lexicon))
    else:
        logger.warning("Lexicon not found at %s", lexicon_path)

    logger.info("All models initialized.")
    yield
    logger.info("Shutting down NoteScanin AI.")


app = FastAPI(
    title="NoteScanin AI Service",
    description="Handwriting OCR API: TrOCR + CRNN + IndoBERT NLP",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(routes.router)
