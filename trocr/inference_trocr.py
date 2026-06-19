import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trocr import config


def load_model(model_dir=config.SAVE_DIR):
    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"TrOCR checkpoint not found: {model_dir}. Train first with "
            "python trocr/train_trocr.py --mode train."
        )

    processor = TrOCRProcessor.from_pretrained(str(model_dir))
    model = VisionEncoderDecoderModel.from_pretrained(
        str(model_dir),
        low_cpu_mem_usage=True,
    )
    model.tie_weights()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return processor, model, device


def predict(image_path, model_dir=config.SAVE_DIR, num_beams=1):
    processor, model, device = load_model(model_dir)
    image = Image.open(image_path).convert("RGB")

    pixel_values = processor(
        image,
        return_tensors="pt",
    ).pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_length=config.MAX_TARGET_LENGTH,
            num_beams=num_beams,
        )

    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Run TrOCR inference on one image.")
    parser.add_argument(
        "image",
        nargs="?",
        default=str(config.LINES_CLEAN_DIR / "0001.png"),
        help="Path to a line image.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(config.SAVE_DIR),
        help="Local TrOCR checkpoint directory.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Beam count for generation. Keep 1 on CPU for speed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(predict(args.image, args.model_dir, args.num_beams))


if __name__ == "__main__":
    main()
