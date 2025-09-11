from transformers import pipeline
import torch

def create_pipeline():
    return pipeline(
        task="text2text-generation",
        model="google-t5/t5-small",
        dtype=torch.float16,
        device=0
    )

def translate(text, source_lang="en", target_lang="fr"):
    prompt = f"translation {source_lang} to {target_lang}: {text}"
    result = create_pipeline()(prompt, max_length=100)
    return result[0]['generated_text']