# ─── Imports ─────────────────────────────────────────────────────────

import io
from typing import Dict, Tuple, List, Union
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile
import json
import base64
import zipfile

# ─── Input functions ───────────────────────────────────────────────── 

def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load raw bytes into a H×W uint8 grayscale array.
    (This preserves the original resolution so you can tile it.)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    return np.array(img)


def load_mask_tif(
    mask_src: Union[str, io.BytesIO]
) -> Tuple[np.ndarray, Dict]:
    with tifffile.TiffFile(mask_src) as tif:
        mask = tif.asarray()
        desc = tif.pages[0].tags["ImageDescription"].value
    metadata = json.loads(desc) if desc else {}
    return mask, metadata

# ─── Output functions ─────────────────────────────────────────────────

def encode_mask_to_tiff_base64(mask: np.ndarray, metadata: Dict) -> Tuple[str, bytes, str]:
    """
    Convert a float mask [0–1] with metadata to base64-encoded TIFF and raw bytes.

    Returns:
      (raw_bytes, base64_str, data_uri)
    """
    # 1) Convert float mask to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)

    # 2) Write to TIFF in memory with metadata
    buf = io.BytesIO()
    tifffile.imwrite(buf, mask_uint8, description=json.dumps(metadata))
    tiff_bytes = buf.getvalue()

    # 3) Base64 encode
    b64 = base64.b64encode(tiff_bytes).decode("utf-8")
    return tiff_bytes, b64, "data:application/octet-stream;base64," + b64