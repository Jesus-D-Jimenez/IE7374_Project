"""Build the submission archive: Group25_FinalProject.zip.

    python scripts/package_submission.py

Every required component sits at the top level under the assignment's naming convention, with a
clean copy of the repository underneath:

    Group25_FinalProject.zip
    ├── Group25_TechnicalReport.pdf      (+ .docx source-of-record)
    ├── Group25_Presentation.pptx
    ├── Group25_Presentation.pdf
    ├── Group25_Presentation.mpeg        <- recorded separately; see --video
    ├── GITHUB_REPOSITORY.txt
    └── IE7374_Project/                  <- every git-tracked file

"Every git-tracked file" is the definition used for the repository copy, so whatever the working
tree ignores (virtualenvs, caches, heavy `results/` artifacts, the binary edit tensors) is
excluded automatically and the archive matches what a reviewer would clone.

The video cannot be generated here — record the talk, then point this script at the file:

    python scripts/package_submission.py --video path/to/recording.mp4

It is copied in under the required name. Without `--video` the archive is still built and the
script warns that the video component is missing.

Usage:
    python scripts/package_submission.py
    python scripts/package_submission.py --video ~/Videos/final.mp4
    python scripts/package_submission.py --output-dir .. --name Group25_FinalProject
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import REPO_ROOT, get_logger                        # noqa: E402

REPO_URL = "https://github.com/Jesus-D-Jimenez/IE7374_Project"
REPO_DIR_IN_ZIP = "IE7374_Project"
REPORT_DIR = os.path.join("docs", "report")

# (path in the repo, name at the top level of the archive) — the graded deliverables, named
# exactly as the assignment requires.
DELIVERABLES = [
    (os.path.join(REPORT_DIR, "Group25_TechnicalReport.pdf"), "Group25_TechnicalReport.pdf"),
    (os.path.join(REPORT_DIR, "Group25_TechnicalReport.docx"), "Group25_TechnicalReport.docx"),
    (os.path.join(REPORT_DIR, "Group25_Presentation.pptx"), "Group25_Presentation.pptx"),
    (os.path.join(REPORT_DIR, "Group25_Presentation.pdf"), "Group25_Presentation.pdf"),
]

VIDEO_NAME = "Group25_Presentation.mpeg"

GITHUB_NOTE = f"""IE7374 Generative AI - Final Project (Milestone 5)
Group 25 - Jesus D. Jimenez Ballestas

GitHub repository:
    {REPO_URL}

The same code is included in this archive under {REPO_DIR_IN_ZIP}/ (every git-tracked file).

Quick start (about a minute on CPU):
    pip install -r requirements.txt
    python src/model_runner.py      # dataset -> model -> outputs/
    pytest -q                       # 38 tests

Read first:
    README.md                       # setup, usage, results, limitations
    outputs/samples.txt             # the generated samples
    outputs/study/README.md         # the six extended experiments behind the report
    docs/report/technical_report.md # the report source (PDF at the top level of this archive)
"""


def tracked_files() -> list[str]:
    """Every git-tracked path, repo-relative — the archive's definition of 'the repository'."""
    out = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, check=True,
                         capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the submission ZIP.")
    ap.add_argument("--output-dir", default="..", dest="output_dir",
                    help="where to write the archive (default: the repository's parent)")
    ap.add_argument("--name", default="Group25_FinalProject",
                    help="archive name without the .zip extension")
    ap.add_argument("--video", default=None,
                    help=f"recorded presentation to include as {VIDEO_NAME} (max 10 minutes)")
    args = ap.parse_args()

    log = get_logger("package")
    out_dir = os.path.abspath(os.path.join(REPO_ROOT, args.output_dir)
                              if not os.path.isabs(args.output_dir) else args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    archive = os.path.join(out_dir, f"{args.name}.zip")

    missing = [src for src, _ in DELIVERABLES if not os.path.exists(os.path.join(REPO_ROOT, src))]
    if missing:
        log.error("missing deliverable(s): %s", ", ".join(missing))
        log.error("build them first: python docs/report/build_report.py && "
                  "python docs/report/build_slides.py")
        return 1

    video = os.path.abspath(args.video) if args.video else None
    if video and not os.path.exists(video):
        log.error("video not found: %s", video)
        return 1

    files = tracked_files()
    log.info("%d tracked files -> %s/", len(files), REPO_DIR_IN_ZIP)

    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in DELIVERABLES:
            zf.write(os.path.join(REPO_ROOT, src), arcname)
            log.info("+ %s", arcname)
        if video:
            zf.write(video, VIDEO_NAME)
            log.info("+ %s (%.1f MB, from %s)", VIDEO_NAME,
                     os.path.getsize(video) / 1e6, video)
            if os.path.splitext(video)[1].lower() not in (".mpeg", ".mpg"):
                log.warning("the source is %s — it is stored under the required .mpeg name but "
                            "the container is unchanged; transcode first if that matters",
                            os.path.splitext(video)[1])
        zf.writestr("GITHUB_REPOSITORY.txt", GITHUB_NOTE)
        log.info("+ GITHUB_REPOSITORY.txt")
        for rel in files:
            source = os.path.join(REPO_ROOT, rel)
            if os.path.exists(source):                 # a tracked-but-deleted file is not fatal
                zf.write(source, f"{REPO_DIR_IN_ZIP}/{rel}")

    size_mb = os.path.getsize(archive) / 1e6
    log.info("wrote %s (%.1f MB, %d entries)", archive, size_mb,
             len(files) + len(DELIVERABLES) + (1 if video else 0) + 1)
    if not video:
        log.warning("no %s in the archive — record the talk (max 10 min), then re-run with "
                    "--video <file>", VIDEO_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
