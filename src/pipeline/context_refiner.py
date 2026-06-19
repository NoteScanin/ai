import os
import logging
import re
import google.generativeai as genai

logger = logging.getLogger("notescanin.nlp_refiner")

def refine_context(ocr_text: str) -> str:
    """
    Advanced NLP Post-Processing untuk merapikan hasil teks OCR berdasarkan konteks kalimat.
    """
    # Tambahkan filter untuk teks kosong untuk mencegah halusinasi
    cleaned = ocr_text.strip()
    if not cleaned:
        return ""
        
    # Jika teks kurang dari 3 karakter atau tidak mengandung huruf sama sekali, kemungkinan besar noise
    if len(cleaned) < 3 or not re.search(r'[a-zA-Z]', cleaned):
        return ocr_text

    api_key = os.getenv("NLP_API_KEY")
    if not api_key:
        logger.warning("NLP_API_KEY tidak ditemukan. Mengembalikan teks asli.")
        return ocr_text

    try:
        genai.configure(api_key=api_key)
        # Menggunakan model bahasa eksternal untuk memperbaiki konteks
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Ini adalah teks hasil scan OCR dari tulisan tangan (bahasa Indonesia). 
        Mungkin ada beberapa kata yang salah eja, typo, atau kurang spasi.
        Tugasmu adalah memperbaiki ejaan yang salah agar menjadi kalimat yang masuk akal dan mudah dibaca sesuai konteks.
        
        Aturan:
        1. Jangan tambahkan penjelasan apapun (seperti "Ini teks yang diperbaiki:").
        2. Kembalikan HANYA teks yang sudah diperbaiki.
        3. Pertahankan format baris (newline) jika ada.
        4. Jika Teks OCR terlihat seperti karakter acak, tidak bermakna, atau hanya noise, kembalikan string kosong.
        
        Teks OCR:
        {ocr_text}
        """
        
        response = model.generate_content(prompt)
        result = response.text.strip()
        
        # Mencegah model merespon dengan string kosong literal (jika dilarang prompt)
        if not result:
            return ""
            
        return result
        
    except Exception as e:
        logger.error(f"Gagal memanggil NLP Refiner: {e}")
        return ocr_text
