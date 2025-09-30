from app.settings import settings
from core.metrics import fails_thresholds

class Router:
    def __init__(self, local_service, hf_service):
        self.local = local_service
        self.hf = hf_service

    def translate(self, text_fr: str) -> dict:
        # 1) essai local
        try:
            text_en, tokens_en, m = self.local.translate(text_fr)
            reasons = fails_thresholds(m, settings)
            if not reasons:
                return {
                    "translation": text_en,
                    "engine": "local",
                    "metrics": m,
                    "fallback_reason": None
                }
            fail = f"local_failed:{','.join(reasons)}"
        except Exception as e:
            fail = f"local_exception:{type(e).__name__}"

        # 2) fallback
        text_en, tokens_en, m = self.hf.translate(text_fr)
        return {
            "translation": text_en,
            "engine": "hf",
            "metrics": m,
            "fallback_reason": fail
        }
