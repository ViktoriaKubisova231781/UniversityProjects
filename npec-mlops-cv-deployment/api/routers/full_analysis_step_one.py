# ─── Imports ────────────────────────────────────────────────

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from pathlib import Path
import io
import zipfile
import numpy as np
from typing import List
from PIL import Image

from app_frontend_final.utils.io_utils import encode_mask_to_tiff_base64
from app_frontend_final.utils.model_utils import load_model, get_model_path_local, call_azure_endpoint
from app_frontend_final.utils.segmentation_utils import segment_image
from app_frontend_final.utils.postprocessing_utils import threshold_mask, morphological_closing

# ─── FastAPI setup ──────────────────────────────────────────

router = APIRouter(prefix="/full_analysis/step_one", tags=["Full Analysis"])

# ─── API endpoint ───────────────────────────────────────────

@router.post("/", summary="Segment image(s) into binary mask(s)")
async def segment_endpoint(
    files: List[UploadFile] = File(...),
    model_id: str = Form(...),
    hosting: str = Form("local")  # 🔧 NEW
):
    if not files:
        raise HTTPException(400, "No files uploaded")

    # 🔧 CLOUD HOSTING
    if hosting == "cloud":
        try:
            image_bytes = await files[0].read()
            return call_azure_endpoint(model_id, image_bytes)
        except Exception as e:
            raise HTTPException(500, f"Azure endpoint call failed: {str(e)}")

    # 🔧 LOCAL HOSTING
    try:
        model_path = get_model_path_local(model_id)
        model = load_model(model_path)
    except Exception as e:
        raise HTTPException(500, f"Model loading failed: {str(e)}")

    mask_files = []

    for file in files:
        try:
            # 1) Validate & read
            if file.content_type not in ("image/png", "image/jpeg", "image/tiff", "image/tif"):
                raise HTTPException(415, f"Unsupported file type: {file.filename}")

            image_bytes = await file.read()
            base = Path(file.filename).stem

            # 2) Run segmentation
            seg_result = segment_image(model, image_bytes)
            raw_mask = seg_result["mask"]

            # 3) Post-process mask
            binary_mask = threshold_mask(raw_mask, 0.1)
            closed_mask = morphological_closing(binary_mask)

            # 4) Encode mask to TIFF with metadata
            metadata = {
                "crop_info": seg_result.get("crop_info"),
                "pad_info": seg_result.get("pad_info")
            }
            tiff_bytes, _, _ = encode_mask_to_tiff_base64(closed_mask, metadata)
            mask_files.append((f"{base}_mask.tif", tiff_bytes))

        except Exception as e:
            raise HTTPException(500, f"Processing error in '{file.filename}': {str(e)}")

    # 5) Return single TIFF or ZIP
    if len(mask_files) == 1:
        name, data = mask_files[0]
        return StreamingResponse(
            io.BytesIO(data),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={name}"}
        )
    else:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zip_out:
            for name, data in mask_files:
                zip_out.writestr(name, data)
        zip_buf.seek(0)

        return StreamingResponse(
            zip_buf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=masks_segmented.zip"}
        )
