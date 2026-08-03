"""Build the project presentation: Markdown -> PPTX -> PDF.

    python docs/report/build_slides.py

pandoc renders `presentation.md` into `Group25_Presentation.pptx` (one slide per `#` heading,
`::: notes` blocks become speaker notes). PowerPoint then exports a PDF copy through COM, so the
deck can be submitted in either format; if PowerPoint is unavailable the PPTX is still produced.

Requires: pandoc on PATH. PowerPoint is optional (PDF stage only).
"""
from __future__ import annotations

import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
SOURCE = os.path.join(HERE, "presentation.md")
PPTX = os.path.join(HERE, "Group25_Presentation.pptx")
PDF = os.path.join(HERE, "Group25_Presentation.pdf")


def markdown_to_pptx() -> str:
    subprocess.run(
        ["pandoc", SOURCE, "--from", "markdown+pipe_tables+fenced_divs",
         "--slide-level=1", "--resource-path", REPO_ROOT, "--output", PPTX],
        cwd=REPO_ROOT, check=True)
    return PPTX


def pptx_to_pdf() -> str | None:
    """Export the deck to PDF through PowerPoint; returns None when it is unavailable."""
    try:
        import win32com.client.dynamic                           # pywin32
    except ImportError:
        print("[pdf] pywin32 not installed — skipping PDF export")
        return None

    app = None
    try:
        # Late binding: PowerPoint's ExportAsFixedFormat rejects the generated wrapper's
        # argument types here, so the deck is saved with SaveAs(ppSaveAsPDF) instead.
        app = win32com.client.dynamic.Dispatch("PowerPoint.Application")
        deck = app.Presentations.Open(PPTX, False, False, False)   # ReadOnly/Untitled/WithWindow
        slides = deck.Slides.Count
        deck.SaveAs(PDF, 32)                                       # 32 = ppSaveAsPDF
        deck.Close()
        print(f"[pdf] {os.path.basename(PDF)} — {slides} slides")
        return PDF
    except Exception as exc:                                      # PowerPoint missing / refused
        print(f"[pdf] PowerPoint export failed ({exc}) — the PPTX is still valid")
        return None
    finally:
        if app is not None:
            app.Quit()


def main() -> int:
    if not shutil.which("pandoc"):
        print("pandoc is required: https://pandoc.org/installing.html")
        return 1
    if not os.path.exists(SOURCE):
        print(f"missing source: {SOURCE}")
        return 1

    markdown_to_pptx()
    print(f"[pptx] {os.path.basename(PPTX)}")
    pptx_to_pdf()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
