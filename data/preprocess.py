"""Preprocessing for the prompt suites.

Because we study a pretrained model rather than training one, "preprocessing" here means
(1) verifying that every answer is a single token so probability comparisons are clean, and
(2) producing reproducible train/validation/test splits for any downstream tuning of the
prompt sets. There is no heavy corpus to clean.
"""
from __future__ import annotations
import random
from typing import Optional


def verify_single_token_answers(model, records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split records into (kept, dropped) by whether both answers are single tokens.

    `model` is a TransformerLens HookedTransformer. A leading space is prepended because
    the GPT-NeoX / GPT-2 BPE tokenizers encode word-initial tokens with a space.
    """
    kept, dropped = [], []
    for r in records:
        answers = [r.get("correct", ""), r.get("incorrect", "")]
        ok = True
        for ans in answers:
            if not ans:
                continue
            toks = model.to_tokens(" " + ans.strip(), prepend_bos=False)
            if toks.shape[1] != 1:
                ok = False
                break
        (kept if ok else dropped).append(r)
    return kept, dropped


def make_splits(
    records: list[dict],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, list[dict]]:
    """Deterministic train/val/test split.

    Contrast pairs are kept together in the same split so a pair is never leaked across
    the train/test boundary.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    by_id = {r["id"]: r for r in records if "id" in r}
    seen, groups = set(), []
    for r in records:
        rid = r.get("id")
        if rid in seen:
            continue
        group = [r]
        seen.add(rid)
        cid = r.get("contrast_id")
        if cid and cid in by_id and cid not in seen:
            group.append(by_id[cid])
            seen.add(cid)
        groups.append(group)

    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    splits = {
        "train": groups[:n_train],
        "val": groups[n_train:n_train + n_val],
        "test": groups[n_train + n_val:],
    }
    return {k: [r for g in v for r in g] for k, v in splits.items()}


if __name__ == "__main__":
    from loaders import load_behavior_suite

    recs = load_behavior_suite("agreement")
    splits = make_splits(recs)
    print({k: len(v) for k, v in splits.items()})
