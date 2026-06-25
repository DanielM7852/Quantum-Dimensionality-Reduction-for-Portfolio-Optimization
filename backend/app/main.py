import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import analyze

app = FastAPI(
    title="EntangleDR Lab API",
    description="Research API for EntangleDR and baseline dimensionality reduction on stock tickers.",
    version="1.0.0",
)

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
