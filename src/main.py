# src/main.py
from typing import List, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import pipeline
import torch

app = FastAPI(title="EN→FR Translation API", version="1.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restreins en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslateENFRRequest(BaseModel):
    text: Union[str, List[str]]
    max_new_tokens: int = Field(128, ge=8, le=512)

class TranslateENFRResponse(BaseModel):
    translations: List[str]

translator = None

@app.on_event("startup")
def load_model():
    global translator
    device = 0 if torch.cuda.is_available() else -1
    # IMPORTANT: text2text-generation pour éviter l'avertissement 'translation_en_to_de'
    translator = pipeline("text2text-generation", model="google-t5/t5-small", device=device)

@app.post("/translate", response_model=TranslateENFRResponse)
def translate(req: TranslateENFRRequest):
    if translator is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    texts: List[str]
    if isinstance(req.text, str):
        texts = [req.text]
    else:
        texts = req.text
        if not texts:
            raise HTTPException(status_code=400, detail="Empty text list")

    # Préfixe T5 explicite pour EN -> FR
    prompts = [f"translate English to French: {t}" for t in texts]

    try:
        outputs = translator(prompts, max_new_tokens=req.max_new_tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    translations = [o.get("generated_text", "") or o.get("translation_text", "") for o in outputs]
    return TranslateENFRResponse(translations=translations)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
