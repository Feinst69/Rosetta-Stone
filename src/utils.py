from pathlib import Path
import unicodedata
from typing import List, Set

# ⇩⇩⇩ NOUVEAU CHEMIN ⇩⇩⇩
NAME_FILE = Path(__file__).resolve().parents[1] / "data" / "raw" / "name.txt"
# ex: Rosetta-Stone/data/raw/name.txt

def _normalize(s: str) -> str:
    """Minuscules + suppression des accents + trim."""
    no_accents = "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )
    return unicodedata.normalize("NFC", no_accents).casefold().strip()

def load_names() -> List[str]:
    """Charge les noms depuis name.txt (UTF-8/UTF-8-BOM toléré)."""
    if not NAME_FILE.exists():
        return []
    names: List[str] = []
    # utf-8-sig tolère un BOM éventuel (éditeurs Windows)
    with open(NAME_FILE, "r", encoding="utf-8-sig") as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)
    return names

def build_index(names: List[str]) -> Set[str]:
    """Set normalisé pour recherche rapide (sans casse/accents)."""
    return {_normalize(n) for n in names}

def name_exists(name: str, index: Set[str]) -> bool:
    """Vrai si 'name' présent (sans casse/accents)."""
    return _normalize(name) in index
