from pydantic import BaseModel

class TranslateReq(BaseModel):
    text: str  # phrase FR

class TranslateResp(BaseModel):
    translation: str
    engine: str
    metrics: dict
    fallback_reason: str | None
