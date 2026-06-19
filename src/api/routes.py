"""
FastAPI route handlers for the OCR service.
"""

from __future__ import annotations

import io
import logging
from collections import Counter

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from src.api.schemas import (
    HealthResponse,
    LineResult,
    ModelStatus,
    OCRResponse,
    RootResponse,
)
from src.inference.crnn import CRNNInference
from src.inference.indobert import IndoBERTCorrector
from src.inference.trocr import TrOCRInference
from src.pipeline.correction import nlp_correct
from src.pipeline.fusion import fuse_predictions
from src.pipeline.preprocessing import preprocess_image
from src.pipeline.segmentation import segment_lines

logger = logging.getLogger("notescanin.api")

router = APIRouter()

# ── Model singletons (set at startup) ────────────────────────────────
trocr: TrOCRInference | None = None
crnn: CRNNInference | None = None
indobert: IndoBERTCorrector | None = None
lexicon: Counter | None = None


def _model_status() -> ModelStatus:
    return ModelStatus(
        trocr=trocr is not None and trocr.is_loaded,
        crnn=crnn is not None and crnn.is_loaded,
        indobert=indobert is not None and indobert.is_loaded,
        lexicon=lexicon is not None,
    )


def _hybrid_predict(line_img: Image.Image) -> tuple[str, float]:
    """Run hybrid OCR on a single line image."""
    trocr_text, trocr_conf = ("", 0.0)
    crnn_text, crnn_conf = ("", 0.0)

    if trocr is not None and trocr.is_loaded:
        trocr_text, trocr_conf = trocr.predict(line_img)

    if crnn is not None and crnn.is_loaded:
        crnn_text, crnn_conf = crnn.predict(line_img)

    # TrOCR Hallucination Guard:
    # If TrOCR generates a sentence that is physically too long for the image
    # and significantly longer than CRNN, it's hallucinating.
    width, height = line_img.size
    aspect_ratio = width / height if height > 0 else 1.0
    max_physical_chars = max(10, int(aspect_ratio * 4) + 5)

    if len(trocr_text) > max(len(crnn_text) * 2.5, len(crnn_text) + 10) and len(trocr_text) > max_physical_chars:
        logger.warning(f"TrOCR halusinasi terdeteksi. CRNN: '{crnn_text}', TrOCR: '{trocr_text}'. Memilih CRNN.")
        return crnn_text, crnn_conf

    # Single model fallback
    if not crnn_text.strip():
        return trocr_text, trocr_conf
    if not trocr_text.strip():
        return crnn_text, crnn_conf

    # Hybrid fusion
    if lexicon is not None:
        try:
            fused = fuse_predictions(crnn_text, trocr_text, lexicon, crnn_conf)
            return fused, (trocr_conf + crnn_conf) / 2
        except Exception:
            pass

    return trocr_text, trocr_conf


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/ocr", response_model=OCRResponse)
async def process_ocr(file: UploadFile = File(...)):
    """
    Full OCR pipeline:
    Image → Preprocessing → Segmentation → Hybrid OCR → NLP Correction
    """
    if (trocr is None or not trocr.is_loaded) and (crnn is None or not crnn.is_loaded):
        raise HTTPException(status_code=503, detail="No OCR model loaded.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 1. Preprocessing
        image = preprocess_image(image)

        # 2. Line segmentation
        line_images = segment_lines(image)
        logger.info("Detected %d lines", len(line_images))

        raw_lines: list[str] = []
        corrected_lines: list[str] = []
        confidences: list[float] = []
        line_details: list[LineResult] = []

        # 3. Per-line OCR + NLP
        for idx, line_img in enumerate(line_images, start=1):
            raw_text, confidence = _hybrid_predict(line_img)

            if not raw_text.strip():
                continue

            clean_text = nlp_correct(raw_text.strip(), indobert, lexicon)

            raw_lines.append(raw_text.strip())
            corrected_lines.append(clean_text)
            confidences.append(confidence)
            line_details.append(LineResult(
                line=idx,
                raw_text=raw_text.strip(),
                clean_text=clean_text,
                confidence=round(confidence, 4),
            ))

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        final_clean_text = "\n".join(corrected_lines)

        return OCRResponse(
            raw_text="\n".join(raw_lines),
            clean_text=final_clean_text,
            confidence=round(avg_conf, 4),
            lines_detected=len(line_details),
            lines=line_details,
            models_used=_model_status(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("OCR pipeline error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Service health check with model status."""
    import torch

    return HealthResponse(
        status="healthy",
        models=_model_status(),
        device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    )


@router.get("/", response_model=RootResponse)
async def root():
    """API information."""
    return RootResponse(
        service="NoteScanin AI",
        version="2.0.0",
        pipeline=[
            "Image Preprocessing",
            "Line Segmentation",
            "Hybrid OCR (TrOCR + CRNN)",
            "NLP Correction (IndoBERT + Lexicon)",
        ],
        endpoints={
            "POST /ocr": "Upload image for OCR",
            "GET /health": "Service health check",
        },
    )
