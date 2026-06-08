# ─── Imports ─────────────────────────────────────────────────────────

import io
from typing import Dict, Tuple, List, Union
import numpy as np
from PIL import Image
import tifffile
import json
from pathlib import Path

# ─── Input functions ───────────────────────────────────────────────── 


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load raw bytes into a H×W uint8 grayscale array.
    (This preserves the original resolution so you can tile it.)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    return np.array(img)


def load_image_file(path: Union[str, Path]) -> np.ndarray:
    with open(path, "rb") as f:
        return load_image_from_bytes(f.read())


def load_mask_tif(mask_src: Union[str, io.BytesIO]) -> Tuple[np.ndarray, Dict]:
    """
    Load a TIFF mask and optional metadata from ImageDescription.
    
    Works even if metadata is missing or malformed (e.g., MNIST test files).
    
    Returns:
        - mask: 2D ndarray
        - metadata: dict (empty if unavailable)
    """
    with tifffile.TiffFile(mask_src) as tif:
        mask = tif.asarray()
        try:
            desc = tif.pages[0].tags["ImageDescription"].value
            metadata = json.loads(desc)
        except (KeyError, json.JSONDecodeError, AttributeError):
            metadata = {}
    return mask, metadata


def get_valid_image_mask_pairs(images_dir: Union[str, Path], masks_dir: Union[str, Path]) -> List[str]:
    """
    Return a sorted list of base filenames that exist in both images and masks.

    Assumes:
      - image name: 'img001.png'
      - mask name: 'img001_root_mask.tif'
    
    Returns:
      List of base names like ['img001', 'img002', ...]
    """
    image_dir = Path(images_dir)
    mask_dir = Path(masks_dir)

    image_basenames = {
        f.stem for f in image_dir.glob("*.png")
    }

    mask_basenames = {
        f.stem.replace("_root_mask", "") for f in mask_dir.glob("*_root_mask.tif")
    }

    valid_pairs = sorted(image_basenames & mask_basenames)
    
    if not valid_pairs:
        raise ValueError("No matching image-mask pairs found.")

    return valid_pairs
