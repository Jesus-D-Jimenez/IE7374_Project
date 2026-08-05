# Technical report and presentation

Both deliverables are written in Markdown and built to their submission formats by a script, so
they can be regenerated from source and reviewed as diffs rather than binaries.

| Source | Build | Output |
|---|---|---|
| [`technical_report.md`](technical_report.md) | `python docs/report/build_report.py` | `Group25_TechnicalReport.docx` + `.pdf` (10 pages) |
| [`presentation.md`](presentation.md) | `python docs/report/build_slides.py` | `Group25_Presentation.pptx` + `.pdf` (20 slides) |
| [`presentation_script.md`](presentation_script.md) | `python docs/report/time_script.py` | the read-aloud narration, timed slide by slide |
| the talk, recorded | — | `Group25_Presentation.mpeg` (max 10 min) |

The video is the one deliverable no script produces. Record the deck being presented — PowerPoint's
*Slide Show → Record* exports video directly — then fold it into the submission archive:

```bash
python scripts/package_submission.py --video path/to/recording.mp4
```

It is stored under the required name; the packaging script warns if the source container is not
actually MPEG. At 10 minutes across 20 slides the budget is ~30 s per slide, so the recording
should follow the speaker notes rather than read the slides.

## Requirements

* **pandoc** on `PATH` (`conda install -c conda-forge pandoc`) — does the Markdown conversion.
* **python-docx** — writes the reference document that carries the report's formatting
  (Times New Roman, 11 pt body, 0.9 in margins, 9.5 pt table text).
* **pywin32 + Microsoft Word / PowerPoint** — optional, used only for the PDF step. Without
  them the DOCX and PPTX are still produced and the script says the PDF was skipped.

The deep-learning stack is **not** needed to build either document. On this machine the report
builds with the Anaconda base interpreter rather than the project `.venv`, because that is where
`pywin32` lives:

```bash
python docs/report/build_report.py     # any interpreter with pandoc + python-docx
python docs/report/build_slides.py
```

## Figures

Every figure is read straight from [`outputs/study/`](../../outputs/study) at build time, so
re-running an experiment and rebuilding the document keeps the two in sync. Image widths are set
per-figure in the Markdown (`{width=4.5in}`) — they are what keeps the report inside the
assignment's 8–10 page budget.

## Notes on the build

Two Word behaviours are worked around in `build_report.py`, both documented at the call site:
pandoc's caption styles carry *keep-with-next*, which was costing about three inches of
whitespace per figure page; and pandoc styles table cells and tight list items with the same
`Compact` style, so table text is shrunk after conversion rather than through the reference
document (which would have dropped bulleted body text below the required 11 pt).

Speaker notes for the talk live in the `::: notes` blocks of `presentation.md` and are carried
into the PPTX notes pane.
