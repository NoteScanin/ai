"""
Centralized configuration for NoteScanin AI Service.
All paths, model settings, and hyperparameters in one place.
"""

import os
import logging
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("notescanin")

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # ai/

# Weights (trained model files)
WEIGHTS_DIR = PROJECT_ROOT / "weights"
CRNN_WEIGHTS_DIR = WEIGHTS_DIR / "crnn"
TROCR_WEIGHTS_DIR = WEIGHTS_DIR / "trocr"
INDOBERT_WEIGHTS_DIR = WEIGHTS_DIR / "indobert"

# Data assets
DATA_DIR = PROJECT_ROOT / "data"
LABELS_DIR = DATA_DIR / "labels"
IMAGES_DIR = DATA_DIR / "images"

# ── CRNN Settings ─────────────────────────────────────────────────────
CRNN_CHECKPOINT = os.getenv(
    "CRNN_CHECKPOINT",
    str(CRNN_WEIGHTS_DIR / "model.pth"),
)
CRNN_VOCAB = os.getenv(
    "CRNN_VOCAB",
    str(CRNN_WEIGHTS_DIR / "vocab.pt"),
)
CRNN_IMG_HEIGHT = 32
CRNN_IMG_WIDTH = 512

# ── TrOCR Settings ───────────────────────────────────────────────────
TROCR_MODEL_DIR = os.getenv(
    "TROCR_MODEL_DIR",
    str(TROCR_WEIGHTS_DIR / "notescan_trocr"),
)
TROCR_FALLBACK_MODEL = "microsoft/trocr-base-handwritten"
TROCR_MAX_LENGTH = 160
TROCR_NUM_BEAMS = 1

# ── IndoBERT Settings ────────────────────────────────────────────────
INDOBERT_MODEL_DIR = os.getenv(
    "INDOBERT_MODEL_DIR",
    str(INDOBERT_WEIGHTS_DIR / "checkpoint-1010"),
)
INDOBERT_MAX_LENGTH = 128
INDOBERT_DECODER_START_TOKEN_ID = 2   # [CLS]
INDOBERT_EOS_TOKEN_ID = 3             # [SEP]
INDOBERT_PAD_TOKEN_ID = 0             # [PAD]

# ── Lexicon ───────────────────────────────────────────────────────────
LEXICON_PATH = os.getenv(
    "LEXICON_PATH",
    str(DATA_DIR / "lexicon.json"),
)

# ── Training Hyperparameters (used by training/ scripts) ─────────────
LABEL_CSV = LABELS_DIR / "label_data.csv"
LINE_IMAGE_DIR = IMAGES_DIR / "segmentation_final"

# CRNN training
CRNN_BATCH_SIZE = 2
CRNN_LEARNING_RATE = 1.0
CRNN_EPOCHS = 30
CRNN_MAX_IMAGE_HEIGHT = 128
CRNN_MAX_LABEL_LEN = 150

# TrOCR training
TROCR_TRAIN_BATCH_SIZE = 1
TROCR_EVAL_BATCH_SIZE = 1
TROCR_GRADIENT_ACCUMULATION = 8
TROCR_LEARNING_RATE = 3e-5
TROCR_EPOCHS = 20
TROCR_MAX_LABEL_CHARS = 150

# Common
VAL_FRACTION = 0.1
TEST_FRACTION = 0.0
SEED = 42
SMOKE_LIMIT = 2
