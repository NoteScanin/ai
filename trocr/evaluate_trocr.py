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
from trocr.train_trocr import (
    character_error_rate,
    configure_model,
    load_approved_dataframe,
    word_error_rate,
)
from ocr.data_quality import split_dataframe


SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained TrOCR model.")
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default="val",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(config.SAVE_DIR),
        help="Local TrOCR checkpoint directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of rows to evaluate. 0 means all rows.",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=5,
        help="Number of prediction examples to print.",
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
    model_dir = Path(args.model_dir)

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
    configure_model(model, processor)
    model.tie_weights()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    result = load_approved_dataframe(config.LABEL_CSV, processor)
    train_df, val_df, test_df = split_dataframe(
        result.dataframe,
        val_fraction=config.VAL_FRACTION,
        test_fraction=config.TEST_FRACTION,
        seed=config.SEED,
    )
    split_map = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
    df = split_map[args.split]

    if df.empty:
        raise ValueError(
            f"Split {args.split!r} is empty. Increase TEST_FRACTION/VAL_FRACTION "
            "or evaluate another split."
        )

    if args.limit > 0:
        df = df.head(args.limit)

    predictions = []
    references = []

    for row in df.to_dict("records"):
        image_path = config.LINES_CLEAN_DIR / row["image"]
        image = Image.open(image_path).convert("RGB")

        pixel_values = processor(
            image,
            return_tensors="pt",
        ).pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_length=config.MAX_TARGET_LENGTH,
                num_beams=args.num_beams,
            )

        prediction = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()

        predictions.append(prediction)
        references.append(row["text"])

    cer = character_error_rate(predictions, references)
    wer = word_error_rate(predictions, references)

    print(f"Split: {args.split}")
    print(f"Samples: {len(predictions)}")
    print(f"CER: {cer:.4f}")
    print(f"WER: {wer:.4f}")

    for index, (prediction, reference) in enumerate(
        zip(predictions, references),
        start=1,
    ):
        if index > args.show_samples:
            break

        print()
        print(f"[{index}] REF: {reference}")
        print(f"[{index}] PRD: {prediction}")


if __name__ == "__main__":
    main()
