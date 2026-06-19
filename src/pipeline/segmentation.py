"""
Page-to-line segmentation pipeline.
Detects individual text lines from a full-page image.
"""

import cv2
import numpy as np
from PIL import Image


def segment_lines(pil_image: Image.Image) -> list[Image.Image]:
    """
    Segment a page image into individual line images using
    horizontal projection profile analysis.

    Returns a list of cropped line images.
    """
    img_gray = np.array(pil_image.convert("L"))
    _, thresh = cv2.threshold(
        img_gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    hist = np.sum(thresh, axis=1)

    lines: list[tuple[int, int]] = []
    in_line = False
    start_y = 0
    threshold_hist = np.max(hist) * 0.05 if np.max(hist) > 0 else 50

    for y, val in enumerate(hist):
        if not in_line and val > threshold_hist:
            in_line = True
            start_y = y
        elif in_line and val <= threshold_hist:
            in_line = False
            if y - start_y > 15:
                margin = 5
                y1 = max(0, start_y - margin)
                y2 = min(img_gray.shape[0], y + margin)
                lines.append((y1, y2))

    if in_line and (img_gray.shape[0] - start_y > 15):
        lines.append((max(0, start_y - 5), img_gray.shape[0]))

    if not lines:
        return [pil_image]

    rgb_image = np.array(pil_image)
    line_images = []
    for y1, y2 in lines:
        if len(rgb_image.shape) == 3:
            crop = rgb_image[y1:y2, :, :]
        else:
            crop = rgb_image[y1:y2, :]
        line_images.append(Image.fromarray(crop))

    return line_images
