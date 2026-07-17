"""Low-level token / probability helpers shared across the package."""
from __future__ import annotations
import math
from collections import Counter

import torch


def first_token(model, s: str) -> int:
    """Token id of the first BPE token of ' s' (GPT-NeoX / GPT-2 prefix a space)."""
    return model.to_tokens(" " + s.strip(), prepend_bos=False)[0, 0].item()


def next_token_logprobs(model, prompt: str) -> torch.Tensor:
    """Log-probabilities over the vocabulary for the token that follows `prompt`."""
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)
    return torch.log_softmax(logits[0, -1], dim=-1)


def top_prediction(model, prompt: str, k: int = 5):
    """Top-k next tokens as (token_string, logprob) pairs."""
    lp = next_token_logprobs(model, prompt)
    vals, idx = lp.topk(k)
    return [(model.to_string(i.item()), round(v.item(), 3)) for v, i in zip(vals, idx)]


def answer_logprob(model, prompt: str, answer: str) -> float:
    return next_token_logprobs(model, prompt)[first_token(model, answer)].item()


def subject_span(model, prompt: str, subject: str) -> tuple[int, int]:
    """(start, end) token positions (inclusive) of `subject` in `prompt`, BOS-offset."""
    str_toks = model.to_str_tokens(prompt, prepend_bos=True)
    text, spans = "", []
    for i, t in enumerate(str_toks):
        if i == 0:            # BOS
            continue
        start = len(text)
        text += t
        end = len(text)
        spans.append((start, end, i))
    ci = text.find(subject)
    if ci == -1:
        ci = text.find(subject.strip())
    if ci == -1:
        raise ValueError(f"subject {subject!r} not found in prompt {prompt!r}")
    cj = ci + len(subject)
    start_pos = next(i for (s, e, i) in spans if e > ci)
    end_pos = next(i for (s, e, i) in spans if e >= cj)
    return start_pos, end_pos


def ngram_entropy(model, text: str, n: int = 3) -> float:
    toks = model.to_str_tokens(text, prepend_bos=False)
    if len(toks) < n:
        return 0.0
    grams = Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    total = sum(grams.values())
    return -sum((c / total) * math.log2(c / total) for c in grams.values())


def fluency(model, text: str) -> float:
    """Mean of bi- and tri-gram entropy; lower => more repetitive / degenerate text."""
    return round((ngram_entropy(model, text, 2) + ngram_entropy(model, text, 3)) / 2, 3)
