# app/main.py

from fastapi import FastAPI
from app_frontend_final.routers.segmentation import router as segmentation_router
from app_frontend_final.routers.root_analysis import router as mask_analysis_router
from app_frontend_final.routers.image_analysis import router as image_analysis_router
from app_frontend_final.routers.full_analysis_step_one import router as full_step_one_router
from app_frontend_final.routers.full_analysis_step_two import router as full_step_two_router
from app_frontend_final.routers.full_analysis_step_three import router as full_step_three_router
from app_frontend_final.routers.full_analysis_step_four import router as full_step_four_router
from app_frontend_final.routers.model_router import router as model_router
from app_frontend_final.routers.feedback_router import router as feedback_router


app = FastAPI(
    title="Plant Root Analysis API",
    description="Modular API for image segmentation, root analysis, and full end-to-end plant processing.",
    version="1.0.0",
)

# ─── Router Mounts ─────────────────────────────────────────

# Individual endpoints
app.include_router(segmentation_router)        # /segment
app.include_router(mask_analysis_router)       # /analyze_mask
app.include_router(image_analysis_router)      # /analyze_image

# Full analysis step-by-step routes
app.include_router(full_step_one_router)       # /full_analysis/step_one
app.include_router(full_step_two_router)       # /full_analysis/step_two
app.include_router(full_step_three_router)     # /full_analysis/step_three
app.include_router(full_step_four_router)      # /full_analysis/step_four

# Model management
app.include_router(model_router)               # /models

# Feedback
app.include_router(feedback_router)            # /feedback

# ─── Health Check ──────────────────────────────────────────

@app.get("/", summary="Health check")
async def health_check():
    return {"status": "ok"}
