# ─── Imports ─────────────────────────────────────────────────────────

import os
import cv2
import numpy as np
from typing import List, Tuple

# ─── Saving functions ────────────────────────────────────────────────

def save_patch_dataset(
    patch_list: List[Tuple[np.ndarray, np.ndarray]],
    output_image_dir: str,
    output_mask_dir: str,
    base_prefix: str = "patch"
) -> None:
    """
    Saves a list of image–mask patch pairs to the specified directories.

    Args:
        patch_list (List): List of (image_patch, mask_patch) tuples.
        output_image_dir (str): Directory to save image patches.
        output_mask_dir (str): Directory to save mask patches.
        base_prefix (str): Prefix used in naming saved files.

    Returns:
        None
    """
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    for idx, (img, mask) in enumerate(patch_list):
        img_path = os.path.join(output_image_dir, f"{base_prefix}_{idx}.png")
        mask_path = os.path.join(output_mask_dir, f"{base_prefix}_{idx}.tif")  # 🔧 changed extension
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)
