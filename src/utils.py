from transformers import pipeline
import torch
import numpy as np
from scipy.special import softmax


def create_pipeline():
    return pipeline(
        task="text2text-generation",
        model="google-t5/t5-small",
        dtype=torch.float16,
        device=0
    )

def t5_translate(text, source_lang="en", target_lang="fr"):
    prompt = f"translation {source_lang} to {target_lang}: {text}"
    result = create_pipeline()(prompt, max_length=100)
    return result[0]['generated_text']

# Critères de fallback basés sur les logits et les tokens générés
def mean_confidence(logits):
    """Moyenne des p_max par pas de temps"""
    probs = softmax(logits, axis=-1)  # (T, vocab)
    p_max = probs.max(axis=-1)        # proba du token choisi à chaque step
    return float(p_max.mean())

def mean_entropy(logits):
    """Moyenne entropie (Shannon) sur la séquence"""
    probs = softmax(logits, axis=-1)
    ent = -np.sum(probs * np.log(probs + 1e-9), axis=-1)
    return float(ent.mean())

def length_ratio(src_tokens, tgt_tokens):
    return (len(tgt_tokens) or 1) / (len(src_tokens) or 1)

def should_fallback(logits, src_tokens, tgt_tokens, eos_token=None, max_len=None):
    reasons = []

    # 1) Confiance
    if mean_confidence(logits) < 0.5:
        reasons.append("low_confidence")

    # 2) Entropie
    if mean_entropy(logits) > 2.5:
        reasons.append("high_entropy")

    # 3) Arrêt prématuré / troncature
    if eos_token is not None:
        try:
            eos_pos = tgt_tokens.index(eos_token)
            if eos_pos < 0.5 * len(tgt_tokens):
                reasons.append("premature_eos")
        except ValueError:
            # pas d’EOS trouvé
            if max_len and len(tgt_tokens) >= max_len:
                reasons.append("truncated_no_eos")

    # 4) Ratio longueur
    ratio = length_ratio(src_tokens, tgt_tokens)
    if not (0.6 <= ratio <= 1.8):
        reasons.append(f"abnormal_length_ratio:{ratio:.2f}")

    return (len(reasons) > 0), reasons
