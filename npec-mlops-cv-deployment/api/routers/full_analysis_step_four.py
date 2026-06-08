from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import tempfile
import os

# Import your report generation helper
from app_frontend_final.utils.analysis_utils import generate_root_analysis_report

router = APIRouter(prefix="/full_analysis/step_four", tags=["Full Analysis"])

@router.post("/", summary="Generate root analysis PDF report from CSV")
async def root_analysis_report(
    csv_file: UploadFile = File(...)
):
    """
    Accepts a CSV of root measurements and returns a downloadable PDF report.
    """
    # Validate file type
    if csv_file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/csv"):  
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {csv_file.content_type}")

    # Create temporary workspace
    tmp_dir = tempfile.mkdtemp()
    try:
        # Save uploaded CSV
        csv_path = os.path.join(tmp_dir, csv_file.filename)
        with open(csv_path, "wb") as f:
            f.write(await csv_file.read())

        # Generate report
        output_pdf = os.path.join(tmp_dir, "root_analysis_report.pdf")
        generate_root_analysis_report(csv_path, output_pdf)

        # Stream PDF back
        pdf_stream = open(output_pdf, "rb")
        return StreamingResponse(
            pdf_stream,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=root_analysis_report.pdf"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp files
        try:
            for fname in os.listdir(tmp_dir):
                os.remove(os.path.join(tmp_dir, fname))
            os.rmdir(tmp_dir)
        except Exception:
            pass
