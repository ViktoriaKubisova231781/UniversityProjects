# ─── Imports ─────────────────────────────────────────────────────────

from patchify import patchify 
import os
import cv2
import numpy as np
from typing import List, Tuple, Dict

# ─── Patchifying functions ───────────────────────────────────────────

def patchify_pair(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 256,
    step: int = 128
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Patchifies the padded image and mask into aligned patches.

    Args:
        image (np.ndarray): Padded grayscale image.
        mask (np.ndarray): Padded binary mask.
        patch_size (int): Size of the patches.
        step (int): Step size between patches.

    Returns:
        List of (image_patch, mask_patch) tuples.
    """
    assert image.shape == mask.shape, "Image and mask must be the same shape before patchifying."

    patches = []

    image_patches = patchify(image, (patch_size, patch_size), step=step)
    mask_patches = patchify(mask, (patch_size, patch_size), step=step)

    for i in range(image_patches.shape[0]):
        for j in range(image_patches.shape[1]):
            img_patch = image_patches[i, j]
            mask_patch = mask_patches[i, j]
            patches.append((img_patch, mask_patch))

    return patches
