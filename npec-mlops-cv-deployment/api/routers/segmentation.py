# ─── Imports ────────────────────────────────────────────────

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pathlib import Path
import io
import zipfile
import numpy as np
import os
import uuid
import json
from typing import List
from PIL import Image

from app_frontend_final.utils.io_utils import encode_mask_to_tiff_base64
from app_frontend_final.utils.model_utils import load_model, get_model_path_local, call_azure_endpoint
from app_frontend_final.utils.segmentation_utils import segment_image
from app_frontend_final.utils.postprocessing_utils import threshold_mask, morphological_closing

# ─── FastAPI setup ──────────────────────────────────────────

router = APIRouter(prefix="/segment", tags=["Segmentation"])

# ─── API endpoint ───────────────────────────────────────────
@router.post("/", summary="Segment image(s) into binary mask(s)")
async def segment_endpoint(
    files: List[UploadFile] = File(...),
    model_id: str = Form(...),
    hosting: str = Form("local")  # 🔧 NEW: "local" or "cloud"
):
    if not files:
        raise HTTPException(400, "No files uploaded")

    # CLOUD model call
    if hosting == "cloud":
        try:
            image_bytes = await files[0].read()
            result = call_azure_endpoint(model_id, image_bytes)
            return result  # You may adapt based on Azure response type
        except Exception as e:
            raise HTTPException(500, f"Azure endpoint call failed: {str(e)}")

    # LOCAL model loading
    try:
        model_path = get_model_path_local(model_id)
        model = load_model(model_path)
    except Exception as e:
        raise HTTPException(500, f"Model loading failed: {str(e)}")
    
    # ─── Session setup ───────────────────────────────────────
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    session_dir = f"./temp_feedback/{session_id}"
    os.makedirs(session_dir, exist_ok=True)

    mask_files = []
    uploaded_filenames = []

    for file in files:
        try:
            if file.content_type not in ("image/png", "image/jpeg", "image/tiff", "image/tif"):
                raise HTTPException(415, f"Unsupported file type: {file.filename}")

            image_bytes = await file.read()
            base = Path(file.filename).stem
            uploaded_filenames.append(file.filename)

            # Save original image for feedback
            with open(f"{session_dir}/{base}.png", "wb") as f:
                f.write(image_bytes)

            # Segment
            seg_result = segment_image(model, image_bytes)
            raw_mask = seg_result["mask"]

            # Post-process
            binary_mask = threshold_mask(raw_mask, 0.1)
            closed_mask = morphological_closing(binary_mask)

            # Encode mask
            metadata = {
                "crop_info": seg_result.get("crop_info"),
                "pad_info": seg_result.get("pad_info")
            }
            tiff_bytes, _, _ = encode_mask_to_tiff_base64(closed_mask, metadata)
            mask_files.append((f"{base}_mask.tif", tiff_bytes))

            # Save mask for feedback
            with open(f"{session_dir}/{base}_mask.tif", "wb") as f:
                f.write(tiff_bytes)

        except Exception as e:
            raise HTTPException(500, f"Processing error in '{file.filename}': {str(e)}")

    # Log session summary
    print(f"[Segment] Saved {len(mask_files)} masks to session {session_id}")

    # Store session ID in frontend (return in header)
    headers = {"X-Session-ID": session_id}

    # Return ZIP
    if len(mask_files) == 1:
        name, data = mask_files[0]
        return StreamingResponse(
            io.BytesIO(data),
            media_type="application/octet-stream",
            headers={**headers, "Content-Disposition": f"attachment; filename={name}"}
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
            headers={**headers, "Content-Disposition": "attachment; filename=masks_segmented.zip"}
        )
