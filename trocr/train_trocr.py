import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Adafactor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trocr import config
from ocr.data_quality import filter_line_level_rows, load_csv_dataframe, split_dataframe


try:
    from multiprocess.resource_tracker import ResourceTracker

    ResourceTracker.__del__ = lambda self: None
except Exception:
    pass


REQUIRED_COLUMNS = {"image", "text", "status"}
APPROVED_STATUS = "approved"


def edit_distance(source, target):
    previous = list(range(len(target) + 1))

    for i, source_item in enumerate(source, start=1):
        current = [i]

        for j, target_item in enumerate(target, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (source_item != target_item)
            current.append(min(insert_cost, delete_cost, replace_cost))

        previous = current

    return previous[-1]


def character_error_rate(predictions, references):
    errors = 0
    total = 0

    for prediction, reference in zip(predictions, references):
        errors += edit_distance(prediction, reference)
        total += len(reference)

    return errors / total if total else 0.0


def word_error_rate(predictions, references):
    errors = 0
    total = 0

    for prediction, reference in zip(predictions, references):
        pred_words = prediction.split()
        ref_words = reference.split()
        errors += edit_distance(pred_words, ref_words)
        total += len(ref_words)

    return errors / total if total else 0.0


def configure_model(model, processor):
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = config.MAX_TARGET_LENGTH
    model.config.num_beams = config.NUM_BEAMS


def freeze_encoder(model):
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False


def preflight_split_csvs(csv_paths):
    for csv_path in csv_paths:
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found. Run dataset/create_review_csv.py, approve "
                "labels, then run dataset/split_dataset.py."
            )

        df = pd.read_csv(csv_path)
        missing_columns = REQUIRED_COLUMNS - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"{csv_path} must be regenerated from approved review labels. "
                f"Missing columns: {', '.join(sorted(missing_columns))}."
            )

        statuses = df["status"].astype(str).str.strip().str.lower()
        approved_count = int((statuses == APPROVED_STATUS).sum())

        if approved_count == 0:
            raise ValueError(
                f"{csv_path} has no approved rows. Approve labels in "
                f"{config.REVIEW_CSV}, then run dataset/split_dataset.py."
            )


@dataclass
class DatasetValidationResult:
    dataframe: pd.DataFrame
    skipped_missing_images: int
    skipped_too_tall: int
    skipped_empty_text: int
    skipped_too_long: int


def load_approved_dataframe(csv_path_or_dataframe, processor, limit=None):
    if isinstance(csv_path_or_dataframe, pd.DataFrame):
        csv_path = None
        df = csv_path_or_dataframe.copy()
        source_name = "dataframe"
    else:
        csv_path = Path(csv_path_or_dataframe)
        df = load_csv_dataframe(csv_path)
        source_name = str(csv_path)

    line_result = filter_line_level_rows(
        df,
        config.LINES_CLEAN_DIR,
        max_image_height=config.MAX_IMAGE_HEIGHT,
        max_label_len=config.MAX_LABEL_CHARS,
        require_approved=True,
        limit=None,
    )

    valid_rows = []
    skipped_token_too_long = 0

    for row in line_result.dataframe.to_dict("records"):
        token_ids = processor.tokenizer(
            row["text"],
            add_special_tokens=True,
            truncation=False,
        ).input_ids

        if len(token_ids) > config.MAX_TARGET_LENGTH:
            skipped_token_too_long += 1
            continue

        valid_rows.append(row)

    valid_df = pd.DataFrame(valid_rows, columns=line_result.dataframe.columns).reset_index(drop=True)

    if limit is not None and limit > 0:
        valid_df = valid_df.head(limit).copy()

    if valid_df.empty:
        raise ValueError(
            f"{source_name} has no usable approved samples. Check labels in "
            f"{config.LABEL_CSV} and keep labels at or below "
            f"{config.MAX_TARGET_LENGTH} tokenizer tokens."
        )

    return DatasetValidationResult(
        dataframe=valid_df,
        skipped_missing_images=line_result.skipped_missing_images,
        skipped_too_tall=line_result.skipped_too_tall,
        skipped_empty_text=line_result.skipped_empty_text,
        skipped_too_long=line_result.skipped_too_long + skipped_token_too_long,
    )


def build_training_args(is_smoke, num_train_epochs, output_dir):
    common_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=1 if is_smoke else config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=num_train_epochs,
        max_steps=1 if is_smoke else -1,
        logging_steps=1 if is_smoke else config.LOGGING_STEPS,
        save_steps=1 if is_smoke else config.SAVE_STEPS,
        eval_steps=1 if is_smoke else config.EVAL_STEPS,
        eval_strategy="no" if is_smoke else "epoch",
        save_strategy="no" if is_smoke else "epoch",
        predict_with_generate=not is_smoke,
        optim="adafactor",
        do_train=True,
        do_eval=not is_smoke,
        fp16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        load_best_model_at_end=False if is_smoke else True,
        metric_for_best_model="cer",
        greater_is_better=False,
        generation_max_length=64 if is_smoke else config.MAX_TARGET_LENGTH,
        generation_num_beams=1,
        save_total_limit=1 if is_smoke else None,
        seed=config.SEED,
    )

    return Seq2SeqTrainingArguments(**common_kwargs)


class OCRDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.df = dataframe.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = config.LINES_CLEAN_DIR / row["image"]

        image = Image.open(image_path).convert("RGB")

        pixel_values = self.processor(
            image,
            return_tensors="pt",
        ).pixel_values.squeeze()

        labels = self.processor.tokenizer(
            row["text"],
            padding="max_length",
            truncation=False,
            max_length=config.MAX_TARGET_LENGTH,
        ).input_ids

        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in labels
        ]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels),
        }


def build_compute_metrics(processor):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_predictions = processor.batch_decode(
            predictions,
            skip_special_tokens=True,
        )

        labels = np.where(
            labels != -100,
            labels,
            processor.tokenizer.pad_token_id,
        )

        decoded_labels = processor.batch_decode(
            labels,
            skip_special_tokens=True,
        )

        decoded_predictions = [text.strip() for text in decoded_predictions]
        decoded_labels = [text.strip() for text in decoded_labels]

        return {
            "cer": character_error_rate(decoded_predictions, decoded_labels),
            "wer": word_error_rate(decoded_predictions, decoded_labels),
        }

    return compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR for NoteScan.")
    parser.add_argument(
        "--mode",
        choices=("smoke", "train"),
        default="train",
        help="Use smoke for a short pipeline check or train for full fine-tuning.",
    )
    parser.add_argument(
        "--model-name",
        default=config.BASE_MODEL_NAME,
        help="Base HuggingFace model or local checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.NUM_TRAIN_EPOCHS,
        help="Number of epochs for train mode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on samples per split. 0 means all rows.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(config.OUTPUT_DIR),
        help="Directory for Trainer checkpoints.",
    )
    parser.add_argument(
        "--save-dir",
        default=str(config.SAVE_DIR),
        help="Directory for the final fine-tuned TrOCR model.",
    )
    parser.add_argument(
        "--no-freeze-encoder",
        dest="freeze_encoder",
        action="store_false",
        help="Allow the TrOCR encoder to train instead of freezing it.",
    )
    parser.set_defaults(freeze_encoder=True)
    return parser.parse_args()


def main():
    args = parse_args()
    is_smoke = args.mode == "smoke"

    print("Loading TrOCR:", args.model_name)
    processor = TrOCRProcessor.from_pretrained(args.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(
        args.model_name,
        low_cpu_mem_usage=True,
    )
    configure_model(model, processor)
    model.tie_weights()

    if args.freeze_encoder:
        freeze_encoder(model)

    smoke_limit = config.SMOKE_LIMIT if is_smoke else None
    effective_limit = smoke_limit if is_smoke else (args.limit if args.limit > 0 else None)
    full_result = load_approved_dataframe(
        config.LABEL_CSV,
        processor,
        limit=effective_limit,
    )
    train_df, val_df, _test_df = split_dataframe(
        full_result.dataframe,
        val_fraction=config.VAL_FRACTION,
        test_fraction=config.TEST_FRACTION,
        seed=config.SEED,
    )

    train_result = load_approved_dataframe(train_df, processor)
    val_result = load_approved_dataframe(val_df, processor)

    print("Train:", len(train_result.dataframe))
    print("Val:", len(val_result.dataframe))

    skipped_missing = full_result.skipped_missing_images
    skipped_too_tall = full_result.skipped_too_tall
    skipped_empty_text = full_result.skipped_empty_text
    skipped_too_long = full_result.skipped_too_long

    if skipped_missing or skipped_too_tall or skipped_empty_text or skipped_too_long:
        print(
            "Skipped samples:",
            f"missing_images={skipped_missing}",
            f"too_tall={skipped_too_tall}",
            f"empty_text={skipped_empty_text}",
            f"too_long={skipped_too_long}",
        )

    train_dataset = OCRDataset(train_result.dataframe, processor)
    val_dataset = OCRDataset(val_result.dataframe, processor)

    sample = train_dataset[0]
    print("Pixel Shape:", tuple(sample["pixel_values"].shape))
    print("Label Shape:", tuple(sample["labels"].shape))

    if is_smoke:
        smoke_batch = default_data_collator([sample])
        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=False,
            relative_step=False,
            lr=config.LEARNING_RATE,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            pixel_values=smoke_batch["pixel_values"],
            labels=smoke_batch["labels"],
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_dir))
        processor.save_pretrained(str(save_dir))

        print("Smoke Loss:", float(loss.item()))
        print(f"Saved checkpoint: {save_dir}")
        print("Smoke selesai")
        return

    training_args = build_training_args(
        is_smoke,
        1 if is_smoke else args.epochs,
        Path(args.output_dir),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        data_collator=default_data_collator,
        compute_metrics=None if is_smoke else build_compute_metrics(processor),
    )

    print("Training Started...")
    trainer.train()

    save_dir = Path(args.save_dir)
    print("Saving model:", save_dir)
    trainer.save_model(str(save_dir))
    processor.save_pretrained(str(save_dir))

    print("Training selesai")


if __name__ == "__main__":
    main()
