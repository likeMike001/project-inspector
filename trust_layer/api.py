"""FastAPI service exposing trust verification metadata."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .verification import build_registry, load_registry, save_registry

app = FastAPI(title="Trust Verification Layer", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
def ensure_registry() -> None:
    if not load_registry():
        save_registry(build_registry())


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/trust/datasets")
def list_datasets() -> dict:
    data = load_registry()
    if not data:
        raise HTTPException(status_code=404, detail="Registry not initialized")
    return {"datasets": data}


@app.post("/trust/verify")
def trigger_verification() -> dict:
    registry = build_registry()
    save_registry(registry)
    return {"datasets": registry}
