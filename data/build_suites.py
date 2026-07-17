"""Generate the behavioral prompt suites and the ROME edit targets.

Every prompt is a *minimal pair*: a `prompt` whose correct next token is `correct`
and whose incorrect alternative is `incorrect`. Each item also carries a `contrast_id`
pointing at the matched item that flips the correct answer, so a behavior can be scored
as a difference of two probabilities rather than a single raw number.

Run from the repo root:
    python data/build_suites.py
This is deterministic; the committed .jsonl / .json files are its output.
"""
from __future__ import annotations
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(HERE, "prompts")


def write_jsonl(name, rows):
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    path = os.path.join(PROMPTS_DIR, name)
    # newline="\n" forces LF on every platform so the committed files are byte-for-byte
    # reproducible (Windows text mode would otherwise write CRLF and break the CI check).
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows):3d} rows -> {os.path.relpath(path, HERE)}")


# --------------------------------------------------------------------------- #
# 1. Subject-verb agreement (with a distractor noun of the opposite number)
# --------------------------------------------------------------------------- #
def build_agreement():
    # (singular subject, plural subject, prepositional-phrase attractor)
    triples = [
        ("The key",       "The keys",       "on the table"),
        ("The author",    "The authors",    "of the book"),
        ("The dog",       "The dogs",       "near the park"),
        ("The student",   "The students",   "in the class"),
        ("The painting",  "The paintings",  "in the gallery"),
        ("The senator",   "The senators",   "from the state"),
        ("The engineer",  "The engineers",  "on the project"),
        ("The child",     "The children",   "at the school"),
        ("The scientist", "The scientists", "in the lab"),
        ("The player",    "The players",    "on the team"),
    ]
    rows, i = [], 0
    for sing, plur, attractor in triples:
        sid, pid = f"agr_{i:03d}_s", f"agr_{i:03d}_p"
        rows.append(dict(id=sid, type="agreement",
                         prompt=f"{sing} {attractor}", correct=" is", incorrect=" are",
                         contrast_id=pid))
        rows.append(dict(id=pid, type="agreement",
                         prompt=f"{plur} {attractor}", correct=" are", incorrect=" is",
                         contrast_id=sid))
        i += 1
    return rows


# --------------------------------------------------------------------------- #
# 2. Factual recall (country/city capitals and a few well-known facts)
# --------------------------------------------------------------------------- #
def build_factual():
    facts = [
        ("The capital of France is",   " Paris",  " London"),
        ("The capital of Japan is",    " Tokyo",  " Beijing"),
        ("The capital of Italy is",    " Rome",   " Madrid"),
        ("The capital of Egypt is",    " Cairo",  " Nairobi"),
        ("The capital of Canada is",   " Ottawa", " Toronto"),
        ("The capital of Russia is",   " Moscow", " Kiev"),
        ("The capital of Spain is",    " Madrid", " Lisbon"),
        ("The capital of Germany is",  " Berlin", " Munich"),
        ("The Eiffel Tower is located in the city of", " Paris", " Rome"),
        ("The Colosseum is located in the city of",    " Rome",  " Athens"),
        ("Mount Everest is located in the country of", " Nepal", " India"),
        ("The Great Barrier Reef is located in",       " Australia", " Brazil"),
    ]
    rows = []
    for i, (prompt, correct, incorrect) in enumerate(facts):
        rows.append(dict(id=f"fact_{i:03d}", type="factual_recall",
                         prompt=prompt, correct=correct, incorrect=incorrect,
                         contrast_id=None))
    return rows


# --------------------------------------------------------------------------- #
# 3. Induction (continue a pattern shown once earlier in the same prompt)
# --------------------------------------------------------------------------- #
def build_induction():
    # "<A> <B> ... <A>" should continue with <B>. Contrast swaps B.
    pairs = [
        ("Alice", "Bob",    "Carol"),
        ("apple", "banana", "cherry"),
        ("north", "south",  "west"),
        ("Monday", "Tuesday", "Friday"),
        ("iron", "copper",  "silver"),
        ("river", "mountain", "desert"),
        ("guitar", "piano", "violin"),
        ("Mercury", "Venus", "Saturn"),
    ]
    rows, i = [], 0
    for a, b, b2 in pairs:
        base = f" {a}{b}"          # a short two-token "fact" to be repeated
        sid, cid = f"ind_{i:03d}_a", f"ind_{i:03d}_b"
        rows.append(dict(id=sid, type="induction",
                         prompt=f"{a}{b} {a}", correct=f"{b}", incorrect=f"{b2}",
                         contrast_id=cid))
        rows.append(dict(id=cid, type="induction",
                         prompt=f"{a}{b2} {a}", correct=f"{b2}", incorrect=f"{b}",
                         contrast_id=sid))
        i += 1
    return rows


# --------------------------------------------------------------------------- #
# 4. Edit targets for the ROME-style generation experiment
# --------------------------------------------------------------------------- #
def build_edit_targets():
    targets = [
        {
            "subject": "The Eiffel Tower",
            "prompt": "The Eiffel Tower is located in the city of",
            "true": "Paris", "new": "Rome",
            "paraphrases": [
                "You can find the Eiffel Tower in the heart of",
                "Tourists who want to see the Eiffel Tower travel to",
            ],
            "neighborhood": [
                ["Big Ben is located in the city of", "London"],
                ["The Brandenburg Gate is located in the city of", "Berlin"],
            ],
        },
        {
            "subject": "Mount Everest",
            "prompt": "Mount Everest is located in the country of",
            "true": "Nepal", "new": "Canada",
            "paraphrases": [
                "To climb Mount Everest you would travel to",
                "Mount Everest can be found in",
            ],
            "neighborhood": [
                ["Mount Fuji is located in the country of", "Japan"],
                ["Kilimanjaro is located in the country of", "Tanzania"],
            ],
        },
        {
            "subject": "The Mona Lisa",
            "prompt": "The Mona Lisa was painted by",
            "true": "Leonardo", "new": "Rembrandt",
            "paraphrases": [
                "The artist who created the Mona Lisa was",
                "The Mona Lisa is a famous work by",
            ],
            "neighborhood": [
                ["The Starry Night was painted by", "Vincent"],
                ["Guernica was painted by", "Pablo"],
            ],
        },
    ]
    path = os.path.join(HERE, "edit_targets.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(targets, f, indent=2)
    print(f"wrote {len(targets):3d} edit targets -> {os.path.relpath(path, HERE)}")


def main():
    write_jsonl("subject_verb_agreement.jsonl", build_agreement())
    write_jsonl("factual_recall.jsonl", build_factual())
    write_jsonl("induction.jsonl", build_induction())
    build_edit_targets()


if __name__ == "__main__":
    main()
