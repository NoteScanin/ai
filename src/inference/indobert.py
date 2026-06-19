"""
IndoBERT typo correction inference module.
Uses fine-tuned EncoderDecoder model for Indonesian text correction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.config import (
    INDOBERT_DECODER_START_TOKEN_ID,
    INDOBERT_EOS_TOKEN_ID,
    INDOBERT_MAX_LENGTH,
    INDOBERT_MODEL_DIR,
    INDOBERT_PAD_TOKEN_ID,
)

logger = logging.getLogger("notescanin.indobert")


class IndoBERTCorrector:
    """Wraps the fine-tuned IndoBERT model for typo correction."""

    def __init__(
        self,
        model_dir: str | None = None,
        device: torch.device | None = None,
    ):
        self._model_dir = Path(model_dir or INDOBERT_MODEL_DIR)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> bool:
        """Load IndoBERT model and tokenizer."""
        if self._loaded:
            return True

        checkpoint = self._resolve_checkpoint()
        if checkpoint is None:
            logger.warning("IndoBERT checkpoint not found. NLP correction disabled.")
            return False

        try:
            from transformers import AutoTokenizer, EncoderDecoderModel

            logger.info("Loading IndoBERT from %s", checkpoint)
            self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
            self.model = EncoderDecoderModel.from_pretrained(
                str(checkpoint), low_cpu_mem_usage=True,
            )
            self.model.config.decoder_start_token_id = INDOBERT_DECODER_START_TOKEN_ID
            self.model.config.eos_token_id = INDOBERT_EOS_TOKEN_ID
            self.model.config.pad_token_id = INDOBERT_PAD_TOKEN_ID

            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info("IndoBERT loaded successfully on %s", self.device)
            return True

        except Exception as e:
            logger.error("Failed to load IndoBERT: %s", e)
            return False

    def _resolve_checkpoint(self) -> Path | None:
        """Find the best available checkpoint directory."""
        if self._model_dir.exists():
            if (self._model_dir / "config.json").exists():
                return self._model_dir
            checkpoints = sorted(self._model_dir.glob("checkpoint-*"), reverse=True)
            if checkpoints:
                return checkpoints[0]
        return None

    def correct(self, text: str) -> str:
        """Correct typos in a single line of text."""
        if not self._loaded or self.model is None:
            return text

        text = text.strip()
        if not text:
            return text

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=INDOBERT_MAX_LENGTH,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=INDOBERT_MAX_LENGTH,
                    decoder_start_token_id=INDOBERT_DECODER_START_TOKEN_ID,
                    eos_token_id=INDOBERT_EOS_TOKEN_ID,
                    pad_token_id=INDOBERT_PAD_TOKEN_ID,
                    repetition_penalty=1.5,
                    no_repeat_ngram_size=3,
                )

            result = self.tokenizer.decode(
                outputs[0].cpu(), skip_special_tokens=True,
            ).strip()

            if not result or len(result) < len(text) * 0.3:
                return text
                
            # Cegah halusinasi: jika hasil jauh lebih panjang dari teks asli
            if len(result) > max(len(text) * 1.5, len(text) + 5):
                logger.warning(f"IndoBERT halusinasi terdeteksi. Asli: '{text}', Hasil: '{result}'. Menggunakan teks asli.")
                return text
                
            return result

        except Exception as e:
            logger.error("IndoBERT correction error: %s", e)
            return text

    def correct_lines(self, text: str) -> str:
        """Correct typos line-by-line for multi-line text."""
        if not self._loaded:
            return text
        lines = text.split("\n")
        return "\n".join(
            self.correct(line) if line.strip() else line
            for line in lines
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded
