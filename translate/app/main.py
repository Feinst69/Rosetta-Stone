# app/main.py
from __future__ import annotations

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# --- Imports projet (assure-toi d'avoir __init__.py dans chaque dossier) ---
from app.schemas import TranslateReq, TranslateResp
from app.settings import settings
from services.local import LocalService
from vendors.hf import HFService
from core.router import Router


# -----------------------------------------------------------------------------
# App & middlewares
# -----------------------------------------------------------------------------
app = FastAPI(title="FR→EN Translator", version="1.0.0")

# CORS (libre en dev; restreins en prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # ex: ["https://ton-domaine.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Static / index.html
# -----------------------------------------------------------------------------
# Dossier 'static' situé à la racine du projet: translate/static/index.html
# Ce fichier (main.py) est dans translate/app/
ROOT_DIR = Path(__file__).resolve().parents[1]     # translate/
STATIC_DIR = ROOT_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static",
    )

@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    if not INDEX_FILE.exists():
        # Si pas d'index.html, renvoyer un message simple
        return JSONResponse({"message": "Index not found. Place your index.html in /static"}, status_code=404)
    return FileResponse(str(INDEX_FILE))


# -----------------------------------------------------------------------------
# Services init (chargés une fois au démarrage)
# -----------------------------------------------------------------------------
# Local LSTM translator (FR->EN)
local_service = LocalService(checkpoint_name="lstm_translator")

# Hugging Face fallback (FR->EN)
hf_service = HFService(model_name=settings.hf_model)

# Router
router = Router(local_service, hf_service)


# -----------------------------------------------------------------------------
# API endpoints
# -----------------------------------------------------------------------------
@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok"}

@app.get("/version", include_in_schema=False)
def version():
    return {"version": app.version}

@app.post("/translate", response_model=TranslateResp)
def translate(req: TranslateReq) -> TranslateResp:
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")
    res = router.translate(text)
    return TranslateResp(**res)


# -----------------------------------------------------------------------------
# Entrée locale (utile en dev)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Lancement local:  uvicorn app.main:app --reload --port 8000
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
