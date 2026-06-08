# ─── Imports ─────────────────────────────────────────────────────────

import cv2
import numpy as np
import random
from typing import List, Tuple
from PIL import Image, ImageEnhance

# ─── Data Augmentation Functions ─────────────────────────────────────

def augment_patches(
    root_rich: List[Tuple[np.ndarray, np.ndarray]]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Applies data augmentation to root-rich patches: flip and enhancement.

    Args:
        root_rich (List of Tuples): List of (image_patch, mask_patch) pairs.

    Returns:
        List of augmented (image_patch, mask_patch) pairs including original,
        flipped, and contrast-enhanced variants.
    """
    augmented = []

    for img, mask in root_rich:
        # Original
        augmented.append((img, mask))

        # Horizontally flipped
        flipped_img = cv2.flip(img, 1)
        flipped_mask = cv2.flip(mask, 1)
        augmented.append((flipped_img, flipped_mask))

        # Enhanced image (mask unchanged)
        pil_img = Image.fromarray(img).convert("L")
        pil_img = ImageEnhance.Brightness(pil_img).enhance(1.2)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.3)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.5)
        enhanced_img = np.array(pil_img)
        augmented.append((enhanced_img, mask))

    return augmented


def sample_and_merge_background(
    clean: List[Tuple[np.ndarray, np.ndarray]],
    noisy: List[Tuple[np.ndarray, np.ndarray]],
    clean_ratio: float,
    noisy_ratio: float,
    target_size: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Samples a balanced background subset and merges clean + noisy patches.

    Args:
        clean (List of Tuples): Clean background patches.
        noisy (List of Tuples): Noisy background patches.
        clean_ratio (float): Ratio of clean patches relative to total target.
        noisy_ratio (float): Ratio of noisy patches relative to total target.
        target_size (int): Base count of root-rich patches to balance against.

    Returns:
        List of sampled background (image_patch, mask_patch) pairs.
    """
    clean_count = int(target_size * clean_ratio)
    noisy_count = int(target_size * noisy_ratio)

    random.shuffle(clean)
    random.shuffle(noisy)

    return clean[:clean_count] + noisy[:noisy_count]