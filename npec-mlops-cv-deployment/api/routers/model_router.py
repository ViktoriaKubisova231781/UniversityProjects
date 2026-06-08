from fastapi import APIRouter
from app_frontend_final.utils.model_utils import list_local_models, list_azure_models

# ─── FastAPI setup ──────────────────────────────────────────

router = APIRouter(prefix="/model_management", tags=["Model Management"])

# ─── API endpoint ───────────────────────────────────────────

@router.get("/models", summary="List local models from /models")
async def get_models():
    return list_local_models()

@router.get("/azure_models", summary="List Azure models from /azure_models")
async def get_azure_models():
    return list_azure_models()