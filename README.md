# NoteScan OCR

Pipeline OCR untuk crop baris tulisan tangan. Project ini sekarang memakai dataset manual terbaru:

- Gambar: `dataset/segmentation_final/*.png`
- Label: `dataset/label_data.csv`
- Format label: `id,label`

`id` otomatis dipetakan ke nama gambar empat digit. Contoh `1` menjadi `0001.png`.

## Struktur

```text
.
+-- crnn/
|   +-- train_crnn.py
|   +-- training.py
|   +-- dataset.py
|   +-- model.py
|   +-- predict.py
|   +-- inference_crnn.py
|   +-- config.py
|   +-- checkpoints/          # checkpoint CRNN per epoch
|   +-- crnn_model.pth        # model CRNN terbaik
|   +-- vocab.pt
|   +-- crnn_metadata.json
+-- trocr/
|   +-- train_trocr.py
|   +-- inference_trocr.py
|   +-- evaluate_trocr.py
|   +-- config.py
|   +-- trocr_model/          # checkpoint Trainer TrOCR per epoch
|   +-- notescan_trocr/       # model TrOCR final
+-- ocr/
|   +-- hybrid_predict.py
|   +-- hybrid.py
|   +-- lexicon.py
|   +-- train_lexicon.py
|   +-- data_quality.py
|   +-- lexicon.json
+-- dataset/
|   +-- segmentation_final/
|   +-- label_data.csv
+-- logs/
```

## Dataset

Loader menerima dua format CSV:

```csv
id,label
1,contoh teks
```

atau format lama:

```csv
image,text,status
0001.png,contoh teks,approved
```

Untuk format `id,label`, semua row dianggap `approved`.

Validasi terakhir pada dataset lokal:

- Label CSV: 503 row.
- Gambar yang cocok: 477 PNG.
- Missing image: 26 row, yaitu `0218.png` sampai `0243.png`.
- Label kosong: 0.
- Label lebih dari 150 karakter: 0.

Training otomatis melewati row yang gambarnya hilang.

## Setup

Windows PowerShell:

```powershell
cd D:\ai
.\.venv\Scripts\Activate.ps1
```

Jika environment perlu dibuat ulang:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pandas pillow numpy opencv-python easyocr transformers accelerate multiprocess safetensors
```

## Training CRNN

Default training CRNN:

- 30 epoch.
- Batch size 2.
- Input width 512 agar CTC cukup panjang untuk label baru.
- Split train/val dibuat deterministik dari `label_data.csv` dengan `VAL_FRACTION=0.1`.
- Checkpoint per epoch disimpan ke `crnn/checkpoints/crnn_epoch_XXX.pth`.
- Model terbaik untuk inferensi disimpan ke `crnn/crnn_model.pth`.

Command:

```powershell
python crnn\train_crnn.py --mode train --epochs 30
```

Smoke test tanpa menimpa model final:

```powershell
python crnn\train_crnn.py --mode smoke `
  --checkpoint tmp\crnn_smoke.pth `
  --vocab tmp\crnn_vocab.pt `
  --metadata tmp\crnn_metadata.json `
  --checkpoint-dir tmp\crnn_checkpoints
```

## Training TrOCR

Default training TrOCR:

- Base model lokal: `trocr/notescan_trocr`.
- 20 epoch.
- Split train/val dibuat deterministik dari `label_data.csv`.
- Checkpoint Trainer disimpan setiap akhir epoch di `trocr/trocr_model`.
- Model final disimpan ke `trocr/notescan_trocr`.

Command:

```powershell
python trocr\train_trocr.py --mode train --epochs 20
```

Smoke test tanpa menimpa model final:

```powershell
python trocr\train_trocr.py --mode smoke `
  --output-dir tmp\trocr_model `
  --save-dir tmp\notescan_trocr_smoke
```

Catatan resource: TrOCR lokal berisi `model.safetensors` sekitar 1.34 GB. Di mesin CPU 8 GB RAM tanpa CUDA, model bisa gagal dimuat jika virtual memory/page file terlalu kecil.

## Lexicon Hybrid

Build ulang lexicon dari dataset baru:

```powershell
python ocr\train_lexicon.py
```

## Inferensi

CRNN:

```powershell
python crnn\inference_crnn.py dataset\segmentation_final\0001.png
```

TrOCR:

```powershell
python trocr\inference_trocr.py dataset\segmentation_final\0001.png
```

Hybrid:

```powershell
python ocr\hybrid_predict.py dataset\segmentation_final\0001.png
```

Full-page note image:

```powershell
python ocr\predict_page.py path\ke\catatan_full.png
```

Output crop baris, teks, dan JSON akan disimpan ke:

```text
outputs/page_ocr/<nama_file>/
```

File penting untuk integrasi backend/web:

- `text.txt`: output utama. Default berisi raw CRNN.
- `raw_text.txt`: hasil CRNN per baris tanpa koreksi lexicon.
- `corrected_text.txt`: alternatif hasil koreksi lexicon jika lexicon aktif.
- `predictions.json`: box, confidence, crop path, raw text, dan corrected text.

Jika ingin `text.txt` berisi hasil koreksi lexicon:

```powershell
python ocr\predict_page.py path\ke\catatan_full.png --primary-output lexicon
```

Kalau halaman punya spiral/jilid di sisi kanan:

```powershell
python ocr\predict_page.py path\ke\catatan_full.png --crop-right-pct 0.96
```

Simpan debug segmentasi:

```powershell
python ocr\predict_page.py path\ke\catatan_full.png --debug
```

CRNN dilatih pada crop baris full-width, jadi horizontal crop dimatikan secara default.
Gunakan `--trim-x` hanya jika halaman upload punya margin kiri/kanan sangat besar.

```powershell
python ocr\predict_page.py path\ke\catatan_full.png --trim-x
```

## Evaluasi TrOCR

Default split evaluasi adalah `val`.

```powershell
python trocr\evaluate_trocr.py --split val
```

Jika `TEST_FRACTION` di `trocr/config.py` masih `0.0`, split `test` kosong.
