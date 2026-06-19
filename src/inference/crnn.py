"""
CRNN inference module.
Includes model architecture and prediction logic.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from src.config import (
    CRNN_CHECKPOINT,
    CRNN_IMG_HEIGHT,
    CRNN_IMG_WIDTH,
    CRNN_VOCAB,
)

logger = logging.getLogger("notescanin.crnn")


# ── Model Architecture ───────────────────────────────────────────────

class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network for text recognition."""

    def __init__(self, num_classes: int, hidden_size=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), stride=(2, 1)),  
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(
            512, hidden_size, bidirectional=True, num_layers=2, batch_first=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)  
        features = self.adaptive_pool(features)  
        features = features.squeeze(2)  
        features = features.permute(0, 2, 1)  
        rnn_out, _ = self.rnn(features)  
        out = self.fc(rnn_out)  
        out = out.permute(1, 0, 2)
        return out


# ── Inference ─────────────────────────────────────────────────────────

class CRNNInference:
    """Handles loading CRNN model and running inference."""

    def __init__(self, checkpoint: str | None = None, vocab: str | None = None):
        self._checkpoint = Path(checkpoint or CRNN_CHECKPOINT)
        self._vocab = Path(vocab or CRNN_VOCAB)
        self.model: CRNN | None = None
        self.idx2char: dict[int, str] = {}
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        self._loaded = False

    def load(self) -> bool:
        """Load CRNN model and vocabulary."""
        if self._loaded:
            return True

        if not self._checkpoint.exists() or not self._vocab.exists():
            logger.warning("CRNN checkpoint or vocab not found, skipping.")
            return False

        try:
            logger.info("Loading CRNN model from %s", self._checkpoint)

            vocab = torch.load(self._vocab, map_location="cpu")
            char2idx = vocab["char2idx"]
            self.idx2char = vocab["idx2char"]

            self.model = CRNN(len(char2idx))
            try:
                checkpoint = torch.load(self._checkpoint, map_location="cpu", mmap=True)
            except TypeError:
                checkpoint = torch.load(self._checkpoint, map_location="cpu")

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]
            self.model.load_state_dict(checkpoint)
            self.model.eval()

            self._loaded = True
            logger.info("CRNN model loaded successfully.")
            return True

        except Exception as e:
            logger.error("Failed to load CRNN: %s", e)
            return False

    def predict(self, image: Image.Image) -> tuple[str, float]:
        """Run CRNN prediction on a PIL image. Returns (text, confidence)."""
        if not self._loaded or self.model is None:
            return "", 0.0

        gray = image.convert("L")
        tensor = self.transform(gray).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor)
            # Model output: [seq_len, batch, classes] → [batch, seq_len, classes]
            logits = logits.permute(1, 0, 2)
            probs = logits.softmax(2)
            best_probs, pred = probs.max(dim=2)

        tokens = pred.squeeze(0).tolist()
        result = []
        prev = None
        for token in tokens:
            if token != prev and token != 0:
                result.append(self.idx2char.get(token, ""))
            prev = token

        text = "".join(result).strip()
        confidence = float(best_probs.squeeze(0).mean().item())
        return text, confidence

    @property
    def is_loaded(self) -> bool:
        return self._loaded
