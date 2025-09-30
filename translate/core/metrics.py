import math

def compute_metrics(pred_ids, per_step_probs, src_len, unk_id=None, eos_id=None):
    # tronquer à EOS si présent
    eff = []
    for i in pred_ids:
        if eos_id is not None and i == eos_id:
            break
        eff.append(i)

    hyp_len = max(1, len(eff))
    len_ratio = hyp_len / max(1, src_len)

    # <unk> rate
    unk_rate = 0.0
    if unk_id is not None and hyp_len > 0:
        unk_cnt = sum(1 for i in eff if i == unk_id)
        unk_rate = unk_cnt / hyp_len

    # log-proba de chaque token choisi
    logps = []
    for t, tok_id in enumerate(eff):
        # per_step_probs[t] est déjà une distribution (softmax) sur V
        p = float(per_step_probs[t][tok_id])
        p = max(p, 1e-12)
        logps.append(math.log(p))

    if not logps:
        avg_logprob = float("-inf")
        cum_logprob = float("-inf")
        tokens_eval = 0
    else:
        cum_logprob = sum(logps)
        avg_logprob = cum_logprob / len(logps)
        tokens_eval = len(logps)

    return {
        "unk_rate": unk_rate,
        "len_ratio": len_ratio,
        "avg_logprob": avg_logprob,
        "cum_logprob": cum_logprob,
        "tokens_evaluated": tokens_eval,
        "hyp_len": hyp_len,
        "src_len": src_len,
    }

def fails_thresholds(m, s) -> list[str]:
    reasons = []
    if m["unk_rate"] > s.unk_rate_max:
        reasons.append("unk_rate")
    if not (s.len_ratio_min <= m["len_ratio"] <= s.len_ratio_max):
        reasons.append("length_ratio")
    if m["avg_logprob"] < s.avg_logprob_min:
        reasons.append("avg_logprob_low")
    # cumul proportionnel à la longueur
    if m["cum_logprob"] < s.cum_logprob_per_token_min * max(1, m["tokens_evaluated"]):
        reasons.append("cum_logprob_low")
    return reasons
