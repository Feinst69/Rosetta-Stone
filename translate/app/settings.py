import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    # Fallback thresholds (sur-définissables par ENV)
    unk_rate_max: float = float(os.getenv("UNK_RATE_MAX", 0.05))
    len_ratio_min: float = float(os.getenv("LEN_RATIO_MIN", 0.7))
    len_ratio_max: float = float(os.getenv("LEN_RATIO_MAX", 1.5))
    avg_logprob_min: float = float(os.getenv("AVG_LOGPROB_MIN", -1.5))
    cum_logprob_per_token_min: float = float(os.getenv("CUM_LOGPROB_PER_TOKEN_MIN", -8.0))

    # Hugging Face model (fallback)
    hf_model: str = os.getenv("HF_MODEL", "Helsinki-NLP/opus-mt-fr-en")

settings = Settings()
