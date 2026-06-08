# ─── Imports ─────────────────────────────────────────────────────────

import numpy as np
from typing import List, Tuple

# ─── Patch Filtering and Classificationfunctions ─────────────────────

def filter_patches_by_mask_content(
    patches: List[Tuple[np.ndarray, np.ndarray]],
    threshold: int = 150
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Splits patch pairs into root-rich and background based on mask content.

    Args:
        patches (List of Tuples): Each tuple is (image_patch, mask_patch).
        threshold (int): Minimum number of white pixels to consider a patch root-rich.

    Returns:
        Tuple of two lists:
            - root_rich: Patches with sufficient root signal.
            - background: Mostly empty/background patches.
    """
    root_rich = []
    background = []

    for img_patch, mask_patch in patches:
        if np.sum(mask_patch >= 1) >= threshold:
            root_rich.append((img_patch, mask_patch))
        else:
            background.append((img_patch, mask_patch))

    return root_rich, background


def classify_background_patches(
    background_patches: List[Tuple[np.ndarray, np.ndarray]],
    std_threshold: float = 4.5
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Further classifies background patches as clean or noisy using image standard deviation.

    Args:
        background_patches (List of Tuples): Patches previously labeled as background.
        std_threshold (float): Standard deviation threshold for noise classification.

    Returns:
        Tuple of two lists:
            - clean: Low-noise background patches.
            - noisy: Background patches with significant noise/variation.
    """
    clean = []
    noisy = []

    for img_patch, mask_patch in background_patches:
        if np.std(img_patch) < std_threshold:
            clean.append((img_patch, mask_patch))
        else:
            noisy.append((img_patch, mask_patch))

    return clean, noisy
