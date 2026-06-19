import os
import csv
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

# Konfigurasi Path
DATA_DIR = "/home/nibras/amikom/NoteScanin/ai/data"
CSV_PATH = os.path.join(DATA_DIR, "labels", "label_data.csv")
IMG_DIR = os.path.join(DATA_DIR, "images", "segmentation_final")
OUTPUT_DIR = "/home/nibras/amikom/NoteScanin/hf-deploy/weights/trocr/notescan_trocr"

class TrOCRDataset(Dataset):
    def __init__(self, data, processor, img_dir):
        self.data = data
        self.processor = processor
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name, text = self.data[idx]
        
        # Load image
        img_path = os.path.join(self.img_dir, file_name)
        if not os.path.exists(img_path):
            image = Image.new('RGB', (100, 32), color = 'white')
        else:
            image = Image.open(img_path).convert("RGB")
        
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Encode text
        labels = self.processor.tokenizer(
            text, padding="max_length", max_length=128
        ).input_ids
        
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        
        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}

def main():
    print("Membaca dataset...")
    with open(CSV_PATH, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        data = [(f"{int(row['id']):04d}.png", row['label']) for row in reader]
    
    random.seed(42)
    random.shuffle(data)
    
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print("Memuat Processor dan Model...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    
    # Configure model
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    # Set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    
    train_dataset = TrOCRDataset(train_data, processor, IMG_DIR)
    val_dataset = TrOCRDataset(val_data, processor, IMG_DIR)
    
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        eval_strategy="epoch",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        fp16=False, # Disable fp16 for CPU
        output_dir=OUTPUT_DIR,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        num_train_epochs=3, # Bisa disesuaikan
        use_cpu=True, # Memaksa penggunaan CPU jika GPU tidak ada
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )
    
    print("Memulai proses training TrOCR (Peringatan: Ini akan memakan waktu lama di CPU)...")
    trainer.train()
    
    print(f"Menyimpan model ke {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Selesai!")

if __name__ == "__main__":
    main()
