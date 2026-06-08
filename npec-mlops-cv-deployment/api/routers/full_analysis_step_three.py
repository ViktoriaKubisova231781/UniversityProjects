# ─── Imports ─────────────────────────────────────────

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from pathlib import Path
import io
import zipfile
from typing import List
from PIL import Image
import numpy as np

from app_frontend_final.utils.model_utils import load_model, get_model_path_local
from app_frontend_final.utils.segmentation_utils import segment_image, segment_plants_from_dish
from app_frontend_final.utils.postprocessing_utils import threshold_mask, morphological_closing, crop_top_and_dish
from app_frontend_final.utils.analysis_utils import (
    measure_primary_root_and_tip,
    adjust_measurements_to_full,
    overlay_roots_on_image
)

# ─── FastAPI setup ──────────────────────────────────

router = APIRouter(prefix="/full_analysis/step_three", tags=["Full Analysis"])

# ─── API endpoint ───────────────────────────────────

@router.post("/", summary="Process image(s) and return image overlays")
async def analyze_image(
    files: List[UploadFile] = File(...),
    model_id: str = Form(...)
):
    if not files:
        raise HTTPException(400, "No files uploaded")

    # Load selected model from /models/
    try:
        model_path = get_model_path_local(model_id)
        model = load_model(model_path)
    except Exception as e:
        raise HTTPException(500, f"Model loading failed: {str(e)}")

    image_overlay_files = []

    for file in files:
        try:
            # 1) Validate input
            if file.content_type not in ("image/png", "image/jpeg", "image/tiff", "image/tif"):
                raise HTTPException(415, f"Unsupported file type: {file.filename}")

            image_bytes = await file.read()
            base = Path(file.filename).stem

            # 2) Segment and refine mask
            seg = segment_image(model, image_bytes)
            raw_mask = seg["mask"]
            binary = threshold_mask(raw_mask, 0.1)
            closed = morphological_closing(binary)
            closed_u8 = (closed * 255).astype(np.uint8)

            # 3) Crop, segment plants
            crop_info = seg.get("crop_info")
            if crop_info is None:
                raise HTTPException(400, f"Image '{file.filename}' is missing crop_info")
            cropped, crop_params = crop_top_and_dish(closed_u8, crop_info)
            segments = segment_plants_from_dish(cropped)

            root_lengths, root_tips, root_paths = {}, {}, {}
            for i, pmask in enumerate(segments, start=1):
                key = f"Plant_{i}"
                length, bot_tip, top_tip, smoothness, angle, depth, span, path = measure_primary_root_and_tip(pmask)
                root_lengths[key] = length
                root_tips[key] = bot_tip
                root_paths[key] = path

            # 4) Overlay on original image
            tips_full, paths_full = adjust_measurements_to_full(root_tips, root_paths, crop_params)
            overlay_img, _ = overlay_roots_on_image(image_bytes, tips_full, paths_full, root_lengths)

            # 5) Save overlay in memory
            buf = io.BytesIO()
            overlay_img.save(buf, format="PNG")
            buf.seek(0)
            image_overlay_files.append((f"{base}_overlay.png", buf.getvalue()))

        except Exception as e:
            raise HTTPException(500, f"Processing error in '{file.filename}': {str(e)}")

    # 6) Return single file or ZIP
    if len(image_overlay_files) == 1:
        name, data = image_overlay_files[0]
        return StreamingResponse(
            io.BytesIO(data),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename={name}"}
        )
    else:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zip_out:
            for name, data in image_overlay_files:
                zip_out.writestr(name, data)
        zip_buf.seek(0)
        return StreamingResponse(
            zip_buf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=image_overlays.zip"}
        )
