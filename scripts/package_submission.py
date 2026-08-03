"""Build the submission archive: Group25_FinalProject.zip.

    python scripts/package_submission.py

The archive contains the three required deliverables at the top level (report PDF, presentation,
GitHub link) plus a clean copy of the repository underneath:

    Group25_FinalProject.zip
    ├── Group25_Technical_Report.pdf
    ├── Group25_Presentation.pptx
    ├── Group25_Presentation.pdf
    ├── GITHUB_REPOSITORY.txt
    └── IE7374_Project/          <- every git-tracked file

"Every git-tracked file" is the definition used for the repository copy, so whatever the working
tree ignores (virtualenvs, caches, heavy `results/` artifacts, the binary edit tensors) is
excluded automatically and the archive matches what a reviewer would clone.

Usage:
    python scripts/package_submission.py
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

# (path in the repo, name at the top level of the archive) — the graded deliverables.
DELIVERABLES = [
    (os.path.join(REPORT_DIR, "Group25_Technical_Report.pdf"), "Group25_Technical_Report.pdf"),
    (os.path.join(REPORT_DIR, "Group25_Technical_Report.docx"), "Group25_Technical_Report.docx"),
    (os.path.join(REPORT_DIR, "Group25_Presentation.pptx"), "Group25_Presentation.pptx"),
    (os.path.join(REPORT_DIR, "Group25_Presentation.pdf"), "Group25_Presentation.pdf"),
]

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

    files = tracked_files()
    log.info("%d tracked files -> %s/", len(files), REPO_DIR_IN_ZIP)

    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in DELIVERABLES:
            zf.write(os.path.join(REPO_ROOT, src), arcname)
            log.info("+ %s", arcname)
        zf.writestr("GITHUB_REPOSITORY.txt", GITHUB_NOTE)
        log.info("+ GITHUB_REPOSITORY.txt")
        for rel in files:
            source = os.path.join(REPO_ROOT, rel)
            if os.path.exists(source):                 # a tracked-but-deleted file is not fatal
                zf.write(source, f"{REPO_DIR_IN_ZIP}/{rel}")

    size_mb = os.path.getsize(archive) / 1e6
    log.info("wrote %s (%.1f MB, %d entries)", archive, size_mb,
             len(files) + len(DELIVERABLES) + 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
