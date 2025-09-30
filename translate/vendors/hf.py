import torch
from transformers import MarianTokenizer, MarianMTModel
from core.metrics import compute_metrics

class HFService:
    def __init__(self, model_name: str, device: str | None = None):
        self.tok = MarianTokenizer.from_pretrained(model_name)
        self.mod = MarianMTModel.from_pretrained(model_name)
        if device and device != "cpu":
            self.mod.to(device)
        self.mod.eval()
        self.device = device

    @torch.no_grad()
    def translate(self, text_fr: str):
        enc = self.tok([text_fr.strip()], return_tensors="pt", padding=True, truncation=True)
        out = self.mod.generate(
            **enc,
            max_length=128,
            return_dict_in_generate=True,
            output_scores=True
        )
        seq = out.sequences[0].tolist()

        # pour Marian, le premier token généré est souvent le token de départ du décodeur
        dec_start = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        gen_ids = seq[1:] if (len(seq) > 0 and seq[0] == dec_start) else seq

        # proba par pas (beam 0 / greedy)
        per_step_probs = []
        for logits in out.scores:
            probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
            per_step_probs.append(probs)

        text_en = self.tok.decode(gen_ids, skip_special_tokens=True)
        out_tokens = [t for t in text_en.strip().split() if t]

        metrics = compute_metrics(
            pred_ids=gen_ids,
            per_step_probs=per_step_probs,
            src_len=len(text_fr.strip().split()),
            unk_id=self.tok.unk_token_id,
            eos_id=self.tok.eos_token_id
        )
        return text_en, out_tokens, metrics
