"""
Image preprocessing pipeline.
Handles contrast enhancement, inversion, and denoising.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


def preprocess_image(pil_image: Image.Image) -> Image.Image:
    """
    Preprocess a handwritten note image for OCR.

    Steps:
        1. Convert to grayscale
        2. Invert if background is dark
        3. Auto-contrast
        4. Boost contrast (2x)
    """
    image_gray = ImageOps.grayscale(pil_image)

    # Invert if background is dark
    if np.mean(np.array(image_gray)) < 127:
        image_gray = ImageOps.invert(image_gray)

    image = image_gray.convert("RGB")
    image = ImageOps.autocontrast(image)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    return image
