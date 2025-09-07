from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from contextlib import asynccontextmanager
import random

# Dossier contenant tes fichiers de noms
NAMES_DIR = Path(__file__).parent / "data" / "names"
ALL_NAMES: list[str] = []

# Fonction utilitaire : charge tous les noms
def load_names() -> list[str]:
    names = []
    if NAMES_DIR.exists():
        for txt in NAMES_DIR.glob("*.txt"):
            with open(txt, "r", encoding="utf-8") as f:
                for line in f:
                    name = line.strip()
                    if name:
                        names.append(name)
    return names or ["Alice", "Bob", "Charlie", "Dora"]  # fallback si vide

# Lifespan (remplace on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ALL_NAMES
    ALL_NAMES = load_names()
    print(f"✅ {len(ALL_NAMES)} noms chargés depuis {NAMES_DIR}")
    yield
    print("👋 Application arrêtée")

# Création de l’app
app = FastAPI(title="Random Name API", lifespan=lifespan)

# Autoriser CORS (dev : * ; en prod -> mettre l’URL de ton front)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles I/O
class AskRequest(BaseModel):
    text: str

class NameResponse(BaseModel):
    name: str

# Endpoint
@app.post("/pick", response_model=NameResponse)
def pick_name(_: AskRequest):
    return NameResponse(name=random.choice(ALL_NAMES))
