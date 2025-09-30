import numpy as np
from core.metrics import compute_metrics

# On utilise ton fichier existant
from LSTM_translator import load_translator_for_inference

class LocalService:
    def __init__(self, checkpoint_name: str = "lstm_translator"):
        # charge le modèle + vocabs une fois
        self.translator = load_translator_for_inference(checkpoint_name)

        # alias utiles
        self.fr2id = self.translator.fr_word_to_idx
        self.en2id = self.translator.en_word_to_idx
        self.id2en = self.translator.en_idx_to_word

        self.pad_id_fr = self.fr2id.get('<pad>', 0)
        self.start_id_en = self.en2id.get('<start>', 1)
        self.end_id_en = self.en2id.get('<end>', None)
        self.unk_id_en = self.en2id.get('<unk>', None)
        self.max_len = self.translator.max_seq_length

    @staticmethod
    def _pad_post(ids, maxlen, pad_id=0):
        ids = (ids or [])[:maxlen]
        return ids + [pad_id] * (maxlen - len(ids))

    def _fr_text_to_tokens(self, s: str) -> list[str]:
        # normalisation minimale, adapte si tu as une pipeline
        return [t for t in s.strip().split() if t]

    def translate(self, text_fr: str):
        fr_tokens = self._fr_text_to_tokens(text_fr)
        fr_ids = [self.fr2id.get(t, self.fr2id.get('<unk>', 1)) for t in fr_tokens]
        src_len = len(fr_tokens)

        # enc input
        enc_in = np.array([self._pad_post(fr_ids, self.max_len, self.pad_id_fr)], dtype='int32')

        # boucle de décodage greedy
        dec_in = np.array([[self.start_id_en]], dtype='int32')
        pred_ids, per_step_probs = [], []

        for _ in range(self.max_len):
            # predict renvoie proba déjà softmaxée sur le vocab EN
            preds = self.translator.model.predict([enc_in, dec_in], verbose=0)  # (1, T, V)
            step_probs = preds[0, -1, :]                                        # (V,)
            tok_id = int(step_probs.argmax())
            pred_ids.append(tok_id)
            per_step_probs.append(step_probs)

            tok = self.id2en.get(tok_id, '<unk>')
            if tok in ('<end>', '<pad>'):
                break

            # préparer le prochain pas
            new_dec = np.zeros((1, dec_in.shape[1] + 1), dtype='int32')
            new_dec[0, :-1] = dec_in[0]
            new_dec[0, -1] = tok_id
            dec_in = new_dec

        # tokens lisibles
        out_tokens = []
        for tid in pred_ids:
            tw = self.id2en.get(tid, '<unk>')
            if tw in ('<start>', '<pad>', '<unk>'):
                continue
            if tw == '<end>':
                break
            out_tokens.append(tw)

        metrics = compute_metrics(
            pred_ids=pred_ids,
            per_step_probs=per_step_probs,
            src_len=src_len,
            unk_id=self.unk_id_en,
            eos_id=self.end_id_en
        )
        return " ".join(out_tokens), out_tokens, metrics
