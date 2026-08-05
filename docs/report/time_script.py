"""Time the presentation script and rewrite its per-slide timestamps.

    python docs/report/time_script.py            # report the timing
    python docs/report/time_script.py --write    # also rewrite the (m:ss-m:ss) markers

The video has a hard 10-minute cap, so the timestamps in `presentation_script.md` have to come
from the words actually written rather than from an estimate made before writing them. This
counts the spoken words under each `## Slide N` heading, converts at `--wpm`, and can write the
cumulative ranges back into the headings so the script stays self-consistent after an edit.

Reading aloud from a script runs 140-160 wpm for most people; 150 is the default and the totals
at 140 and 160 are printed so the margin against the cap is visible.
"""
from __future__ import annotations

import argparse
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "presentation_script.md")
CAP_SECONDS = 10 * 60

SLIDE_RE = re.compile(r"^## Slide (\d+) — (.+?)(?: \*\(\d+:\d\d–\d+:\d\d\)\*)?$", re.M)


def mmss(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def spoken_words(text: str) -> int:
    """Words a presenter actually says: no markdown emphasis, no blockquote markers."""
    return len(re.sub(r"[*_`>]", "", text).split())


def main() -> int:
    ap = argparse.ArgumentParser(description="Time the presentation script.")
    ap.add_argument("--wpm", type=float, default=150.0, help="speaking pace (default 150)")
    ap.add_argument("--write", action="store_true",
                    help="rewrite the (m:ss-m:ss) markers in the slide headings")
    args = ap.parse_args()

    text = open(SCRIPT, encoding="utf-8").read()
    narration = text.split("# Q&A preparation")[0]

    headings = list(SLIDE_RE.finditer(narration))
    if not headings:
        print(f"no slide headings found in {SCRIPT}")
        return 1

    counts = []
    for i, match in enumerate(headings):
        start = match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(narration)
        counts.append(spoken_words(narration[start:end]))

    print(f"{'slide':>5}  {'words':>5}  {'seconds':>7}  running")
    elapsed, ranges = 0.0, []
    for match, words in zip(headings, counts):
        seconds = words / args.wpm * 60
        ranges.append((elapsed, elapsed + seconds))
        elapsed += seconds
        print(f"{match.group(1):>5}  {words:>5}  {seconds:>6.0f}s  {mmss(elapsed)}")

    total = sum(counts)
    print(f"\n{total} spoken words")
    for wpm in (140, args.wpm, 160):
        seconds = total / wpm * 60
        flag = "  OVER THE 10-MINUTE CAP" if seconds > CAP_SECONDS else ""
        print(f"  {wpm:g} wpm -> {mmss(seconds)}{flag}")

    if args.write:
        def replace(match: re.Match) -> str:
            index = int(match.group(1)) - 1
            start, end = ranges[index]
            return f"## Slide {match.group(1)} — {match.group(2)} *({mmss(start)}–{mmss(end)})*"

        updated = SLIDE_RE.sub(replace, narration) + text.split("# Q&A preparation", 1)[1] \
            .join(["# Q&A preparation", ""])
        with open(SCRIPT, "w", encoding="utf-8", newline="\n") as f:
            f.write(updated)
        print(f"\nrewrote the timestamps in {os.path.basename(SCRIPT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
