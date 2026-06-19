"""
TrOCR inference module.
Handles model loading and handwriting recognition.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image

from src.config import (
    TROCR_FALLBACK_MODEL,
    TROCR_MAX_LENGTH,
    TROCR_MODEL_DIR,
)

logger = logging.getLogger("notescanin.trocr")


class TrOCRInference:
    """Handles loading TrOCR model and running OCR inference."""

    def __init__(self, model_dir: str | None = None, device: torch.device | None = None):
        self._model_dir = model_dir or TROCR_MODEL_DIR
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        self._loaded = False

    def load(self) -> bool:
        """Load TrOCR processor and model."""
        if self._loaded:
            return True

        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            model_source = self._model_dir
            model_path = Path(self._model_dir)
            has_weights = (
                model_path.exists()
                and (
                    (model_path / "pytorch_model.bin").exists()
                    or (model_path / "model.safetensors").exists()
                )
            )
            if not has_weights:
                logger.warning(
                    "Local TrOCR weights not found at %s, falling back to %s",
                    self._model_dir, TROCR_FALLBACK_MODEL,
                )
                model_source = TROCR_FALLBACK_MODEL

            logger.info("Loading TrOCR from %s", model_source)
            self.processor = TrOCRProcessor.from_pretrained(model_source)
            self.model = VisionEncoderDecoderModel.from_pretrained(
                model_source, low_cpu_mem_usage=True,
            ).to(self.device)
            self.model.eval()

            self._loaded = True
            logger.info("TrOCR loaded successfully on %s", self.device)
            return True

        except Exception as e:
            logger.error("Failed to load TrOCR: %s", e)
            return False

    def predict(self, image: Image.Image) -> tuple[str, float]:
        """
        Run TrOCR on a single line image.
        Returns (text, confidence).
        """
        if not self._loaded or self.model is None or self.processor is None:
            return "", 0.0

        pixel_values = self.processor(
            images=image, return_tensors="pt",
        ).pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                max_length=TROCR_MAX_LENGTH,
                repetition_penalty=1.5,
                no_repeat_ngram_size=4,
            )

        generated_ids = outputs.sequences
        text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True,
        )[0].strip()

        # Calculate confidence from token probabilities
        confidence = 0.95
        if outputs.scores:
            probs = []
            for i, score in enumerate(outputs.scores):
                token_probs = torch.nn.functional.softmax(score, dim=-1)
                if i + 1 < len(generated_ids[0]):
                    token_id = generated_ids[0][i + 1]
                    prob = token_probs[0, token_id].item()
                    probs.append(prob)
            if probs:
                confidence = sum(probs) / len(probs)

        return text, confidence

    @property
    def is_loaded(self) -> bool:
        return self._loaded
