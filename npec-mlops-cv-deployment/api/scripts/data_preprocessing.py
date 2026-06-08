# ─── Imports ─────────────────────────────────────────────────────────

import cv2
import numpy as np
from typing import Tuple, Dict

# ─── Preprocessing functions ─────────────────────────────────────────

def cropper(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Crops the image and mask around the largest contour found in the image.
    Ensures the crop is square and centered around the largest region.
    Returns the cropped image, cropped mask, and crop metadata.
    """
    orig_shape = image.shape

    # Blur and threshold
    blurred_image = cv2.GaussianBlur(image, (11, 11), 0)
    _, otsu_thresh = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(otsu_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, mask, {"original_shape": orig_shape, "used_crop": False}

    # Largest contour → square crop
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    size = max(w, h)
    x_center = x + w // 2
    y_center = y + h // 2
    x_start = max(0, x_center - size // 2)
    y_start = max(0, y_center - size // 2)

    # Crop both
    cropped_image = image[y_start:y_start + size, x_start:x_start + size]
    cropped_mask = mask[y_start:y_start + size, x_start:x_start + size]

    crop_info = {
        "original_shape": orig_shape,
        "used_crop": True,
        "x_start": x_start,
        "y_start": y_start,
        "crop_size": size
    }

    return cropped_image, cropped_mask, crop_info


def padder(image: np.ndarray, mask: np.ndarray, patch_size: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Pads the image and mask to be divisible by the patch size.
    Also returns the padding info (top, bottom, left, right).
    """
    h, w = image.shape[:2]

    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding  = ((w // patch_size) + 1) * patch_size - w

    top = height_padding // 2
    bottom = height_padding - top
    left = width_padding // 2
    right = width_padding - left

    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=0)
    padded_mask = cv2.copyMakeBorder(mask, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=0)

    pad_info = {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right
    }

    return padded_image, padded_mask, pad_info

