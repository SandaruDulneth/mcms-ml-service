from fastapi import APIRouter

from app.api.routes import extract, info, predict


api_router = APIRouter()
api_router.include_router(info.router)
api_router.include_router(predict.router)
api_router.include_router(extract.router)
