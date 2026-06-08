# ─── Imports ─────────────────────────────────────────

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from pathlib import Path
import io
import zipfile
from typing import List
from PIL import Image
import numpy as np

from app_frontend_final.utils.model_utils import load_model, get_model_path_local, call_azure_endpoint
from app_frontend_final.utils.segmentation_utils import segment_image, segment_plants_from_dish
from app_frontend_final.utils.postprocessing_utils import threshold_mask, morphological_closing, crop_top_and_dish
from app_frontend_final.utils.analysis_utils import (
    measure_primary_root_and_tip,
    adjust_measurements_to_full,
    overlay_roots_on_image,
    build_root_measurement_csv
)

# ─── FastAPI setup ──────────────────────────────────

router = APIRouter(prefix="/analyze_image", tags=["Image Analysis"])

# ─── API endpoint ───────────────────────────────────
@router.post("/", summary="Process image(s) and return overlays + measurements as ZIP")
async def analyze_image(
    files: List[UploadFile] = File(...),
    model_id: str = Form(...),
    hosting: str = Form("local")  # 🔧 NEW
):
    if not files:
        raise HTTPException(400, "No files uploaded")

    # 🔧 CLOUD MODEL LOGIC
    if hosting == "cloud":
        try:
            image_bytes = await files[0].read()
            return call_azure_endpoint(model_id, image_bytes)  # Your Azure logic or placeholder
        except Exception as e:
            raise HTTPException(500, f"Azure endpoint call failed: {str(e)}")

    # 🔧 LOCAL MODEL LOADING
    try:
        model_path = get_model_path_local(model_id)
        model = load_model(model_path)
    except Exception as e:
        raise HTTPException(500, f"Model loading failed: {str(e)}")

    image_overlay_files = []
    csv_data = []

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
                csv_data.append({
                    "filename": file.filename,
                    "plant": key,
                    "length": length,
                    "bottom_tip": bot_tip,
                    "top_tip": top_tip,
                    "smoothness": smoothness,
                    "angle": angle,
                    "depth": depth,
                    "span": span
                })

            # 4) Overlay on original image
            tips_full, paths_full = adjust_measurements_to_full(root_tips, root_paths, crop_params)
            overlay_img, _ = overlay_roots_on_image(image_bytes, tips_full, paths_full, root_lengths)

            # 5) Save overlay to memory
            buf = io.BytesIO()
            overlay_img.save(buf, format="PNG")
            buf.seek(0)
            image_overlay_files.append((f"{base}_overlay.png", buf.getvalue()))

        except Exception as e:
            raise HTTPException(500, f"Processing error in '{file.filename}': {str(e)}")

    # 6) Save CSV
    csv_name, csv_bytes = build_root_measurement_csv(csv_data)

    # 7) Create final ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zip_out:
        if len(files) == 1:
            zip_out.writestr(image_overlay_files[0][0], image_overlay_files[0][1])
            zip_out.writestr(csv_name, csv_bytes)
        else:
            for name, data in image_overlay_files:
                zip_out.writestr(f"image_overlays/{name}", data)
            zip_out.writestr(csv_name, csv_bytes)
    zip_buf.seek(0)

    # 8) Return download
    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=image_analysis_results.zip"}
    )
