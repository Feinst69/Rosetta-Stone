from contextlib import asynccontextmanager
from typing import List, Set

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .utils import load_names, build_index, name_exists, NAME_FILE

ALL_NAMES: List[str] = []
NAMES_INDEX: Set[str] = set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ALL_NAMES, NAMES_INDEX
    ALL_NAMES = load_names()
    NAMES_INDEX = build_index(ALL_NAMES)
    print(f"✅ {len(ALL_NAMES)} noms chargés depuis {NAME_FILE}")
    yield
    print("👋 App arrêtée")

app = FastAPI(title="Name Checker API", lifespan=lifespan)

# CORS (ouvert en dev ; en prod → restreindre à ton domaine front)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NameRequest(BaseModel):
    name: str

class CheckResponse(BaseModel):
    exists: bool

@app.post("/check", response_model=CheckResponse)
def check_name(req: NameRequest) -> CheckResponse:
    """Vérifie si le nom est présent dans data/raw/name.txt."""
    return CheckResponse(exists=name_exists(req.name, NAMES_INDEX))

@app.get("/health")
def health():
    return {"ok": True, "count": len(ALL_NAMES)}
