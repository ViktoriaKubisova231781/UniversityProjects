# ─── Imports ────────────────────────────────────────────────

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pathlib import Path
import io
import zipfile
from typing import List
from PIL import Image
import numpy as np

from app_frontend_final.utils.io_utils import load_mask_tif
from app_frontend_final.utils.postprocessing_utils import crop_top_and_dish
from app_frontend_final.utils.segmentation_utils import (
    segment_plants_from_dish,
    merge_segmented_masks,
    reconstruct_full_mask
)
from app_frontend_final.utils.analysis_utils import (
    measure_primary_root_and_tip,
    adjust_measurements_to_full,
    render_full_mask_with_roots_tiff,
    build_root_measurement_csv
)

# ─── FastAPI setup ──────────────────────────────────────────

router = APIRouter(prefix="/full_analysis/step_two", tags=["Full Analysis"])

# ─── API endpoint ───────────────────────────────────────────

@router.post("/", summary="Process one or more mask TIFFs and return overlays + measurements")
async def process_mask(
    files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(400, "No files uploaded")

    overlay_files = []
    csv_data = []

    for file in files:
        try:
            # 1) Validate & read
            if file.content_type not in ("image/tiff", "image/tif"):
                raise HTTPException(415, f"Unsupported file type: {file.filename}")

            data = await file.read()
            buf = io.BytesIO(data)

            # 2) Load mask + metadata
            mask_full, full_meta = load_mask_tif(buf)
            crop_info = full_meta.get("crop_info")
            if crop_info is None:
                raise HTTPException(400, f"TIFF '{file.filename}' is missing 'crop_info' metadata")

            # 3) Crop, segment, measure
            cropped, crop_params = crop_top_and_dish(mask_full, crop_info)
            segmented = segment_plants_from_dish(cropped)

            root_lengths, root_tips, root_paths = {}, {}, {}
            base = Path(file.filename).stem
            for i, pm in enumerate(segmented, start=1):
                key = f"Plant_{i}"
                length, bot_tip, top_tip, smoothness, angle, depth, span, path = measure_primary_root_and_tip(pm)
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

            # 4) Reconstruct and overlay
            merged_crop = merge_segmented_masks(segmented, crop_params)
            full_mask = reconstruct_full_mask(merged_crop, crop_params)
            tips_full, paths_full = adjust_measurements_to_full(root_tips, root_paths, crop_params)
            _, overlay_tiff = render_full_mask_with_roots_tiff(full_mask, root_lengths, tips_full, paths_full)

            # 5) Add overlay file
            overlay_files.append((base, overlay_tiff))

        except Exception as e:
            raise HTTPException(500, f"Processing error in '{file.filename}': {str(e)}")

    # 6) Build CSV
    csv_name, csv_bytes = build_root_measurement_csv(csv_data)

    # 7) Create ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zip_out:

        if len(overlay_files) == 1:
            # One file → root-level CSV and TIF
            base, tif = overlay_files[0]
            zip_out.writestr(f"{base}_overlay.tif", tif)
            zip_out.writestr(csv_name, csv_bytes)
        else:
            # Multiple → overlays/ folder and root-level CSV
            for base, tif in overlay_files:
                zip_out.writestr(f"mask_overlays/{base}_overlay.tif", tif)
            zip_out.writestr(csv_name, csv_bytes)

    zip_buf.seek(0)

    # 8) Return ZIP file for download
    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=mask_analysis_results.zip"}
    )
