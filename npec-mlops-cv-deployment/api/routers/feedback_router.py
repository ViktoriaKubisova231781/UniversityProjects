# ─── Imports ────────────────────────────────────────────────

from fastapi import APIRouter, HTTPException, Form, Query
from fastapi.responses import StreamingResponse
from typing import List
from pathlib import Path
import zipfile, io, shutil, subprocess
import io
import json

# ─── Router Setup ───────────────────────────────────────────

router = APIRouter(prefix="/feedback", tags=["Feedback"])

# ─── 1. Get 10 Image-Mask Pairs for Feedback UI ─────────────

@router.get("/session/{session_id}")
def get_feedback_images(session_id: str):
    session_path = Path(f"./temp_feedback/{session_id}")
    if not session_path.exists():
        raise HTTPException(404, "Session not found")

    image_mask_pairs = []
    all_files = sorted(f.stem for f in session_path.glob("*.png"))[:10]

    for base in all_files:
        image_mask_pairs.append({
            "base": base,
            "image_url": f"/feedback/image?type=image&session_id={session_id}&file={base}.png",
            "mask_url": f"/feedback/image?type=mask&session_id={session_id}&file={base}_mask.tif"
        })

    return image_mask_pairs

# ─── 2. Serve Static Images & Masks for UI ──────────────────


@router.get("/image")
def serve_image(
    type: str = Query(...),
    session_id: str = Query(...),
    file: str = Query(...)
):
    folder = Path(f"./temp_feedback/{session_id}")
    filepath = folder / file

    if not filepath.exists():
        raise HTTPException(404, "File not found")

    media_type = "image/tiff" if type == "mask" else "image/png"
    return StreamingResponse(open(filepath, "rb"), media_type=media_type)

# ─── 3. Submit Feedback & Save Correct Pairs ────────────────


@router.post("/submit")
async def submit_feedback(
    feedback_raw: str = Form(...),  # ← Frontend sends JSON string
    session_id: str = Form(...)
):
    try:
        feedback = json.loads(feedback_raw)
    except json.JSONDecodeError:
        raise HTTPException(400, detail="Invalid feedback JSON")

    correct_dir = Path("./feedback_data")
    incorrect_buf = io.BytesIO()
    session_path = Path(f"./temp_feedback/{session_id}")
    correct_dir.mkdir(exist_ok=True)

    incorrect_names = []

    # Save correct and collect incorrect
    for item in feedback:
        base = item["base"]
        if item["status"] == "correct":
            shutil.copy(session_path / f"{base}.png", correct_dir / f"{base}.png")
            shutil.copy(session_path / f"{base}_mask.tif", correct_dir / f"{base}_mask.tif")
        else:
            incorrect_names.append(base)

    # Bundle incorrect into ZIP
    with zipfile.ZipFile(incorrect_buf, "w", zipfile.ZIP_DEFLATED) as zip_out:
        for base in incorrect_names:
            zip_out.write(session_path / f"{base}.png", arcname=f"{base}.png")
            zip_out.write(session_path / f"{base}_mask.tif", arcname=f"{base}_mask.tif")
    incorrect_buf.seek(0)

    # Trigger retraining if enough new feedback
    correct_files = list(correct_dir.glob("*.png"))
    if len(correct_files) >= 20:
        print("Triggering retraining pipeline...")
        try:
            subprocess.run(["python", "training_scripts/preprocess_feedback.py"], check=True)
            subprocess.run(["python", "training_scripts/train_model.py"], check=True)
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"Retraining failed: {e}")

    return StreamingResponse(
        incorrect_buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=incorrect_feedback.zip"}
    )
