"""End-to-end pipeline: preprocessed dataset -> pretrained generative model -> saved samples.

This is the single entry point for the milestone:

    python src/model_runner.py

which will

  1. load (building it first if necessary) the processed dataset in `data/processed/`;
  2. load the pretrained model named in `configs/model_config.yaml` (default `pythia-160m`);
  3. run inference on a balanced batch of 10 behavioral prompts — greedy continuation,
     top-k next tokens, and the correct-vs-incorrect logit difference;
  4. run the project's generative experiment on every ROME edit target — generate before
     the edit, fit a rank-one edit to the localized fact, generate again, and score
     efficacy / generalization / specificity / fluency;
  5. write every generation to `outputs/` as human-readable text, CSV tables, a summary
     figure, and a JSON run record.

Useful variations:

    python src/model_runner.py --skip-edit            # fast: behavioral samples only (~1 min)
    python src/model_runner.py --n-samples 12         # more prompts
    python src/model_runner.py --model gpt2           # the replication control model
    python src/model_runner.py --edit-layer 5         # skip causal tracing, force the layer
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

# Make the repository root importable when this file is run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import ProjectConfig, load_config                      # noqa: E402
from src.data_loader import (                                             # noqa: E402
    load_manifest, load_processed_edit_targets, load_processed_prompts,
    select_inference_samples,
)
from utils.helpers import (                                               # noqa: E402
    banner, ensure_dir, get_logger, one_line, relative_to_repo, set_seed,
    write_json, write_text,
)
from utils.visualization import bar_chart, grouped_bar_chart              # noqa: E402

# Output file names — one convention, documented in outputs/README.md.
SAMPLES_TXT = "samples.txt"
BEHAVIORAL_CSV = "samples_behavioral.csv"
EDIT_CSV = "samples_edit.csv"
EDIT_SUMMARY_CSV = "edit_summary.csv"
RUN_METADATA_JSON = "run_metadata.json"
IMAGES_SUBDIR = "images"

_INSTALL_HINT = (
    "The deep-learning stack is not installed in this interpreter.\n"
    "    pip install -r requirements.txt\n"
    "or run the container:  docker build -t black-box-lm . && docker run --rm black-box-lm"
)


class PipelineRunner:
    """Owns the loaded model and turns dataset records into scored, saved samples."""

    def __init__(self, config: Optional[ProjectConfig] = None, logger=None):
        self.cfg = config or load_config()
        self.log = logger or get_logger("model_runner")
        self._model = None

    # ----------------------------------------------------------------- model
    @property
    def model(self):
        """Load the pretrained model once, with a readable error if deps are missing."""
        if self._model is None:
            try:
                from models.loader import load_model
            except ImportError as exc:                      # torch / transformer_lens absent
                raise SystemExit(f"{exc}\n\n{_INSTALL_HINT}") from exc

            self.log.info("loading %s (first run downloads ~380 MB from Hugging Face)",
                          self.cfg.model_name)
            started = time.time()
            try:
                self._model = load_model(self.cfg)
            except OSError as exc:                          # download / cache failure
                raise SystemExit(
                    f"could not load '{self.cfg.model_name}': {exc}\n"
                    "Check the network connection, or point HF_HOME at a warm cache."
                ) from exc
            self.log.info("model ready in %.1fs on %s", time.time() - started,
                          self.cfg.resolved_device())
        return self._model

    def model_facts(self) -> dict:
        """Architecture summary recorded in the run metadata."""
        cfg = self.model.cfg
        return {
            "name": self.cfg.model_name,
            "device": self.cfg.resolved_device(),
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "d_model": cfg.d_model,
            "d_mlp": cfg.d_mlp,
            "n_params": int(sum(p.numel() for p in self.model.parameters())),
        }

    # ------------------------------------------------------- behavioral pass
    def score_prompt(self, record: dict) -> dict:
        """Run one prompt through the model and return every number we report for it.

        A single forward pass supplies the whole row: the log-softmax difference between
        the correct and incorrect answer equals the raw logit difference (the shared
        normalizer cancels), so there is no need to re-run the model per metric.
        """
        from models.generation import generate
        from models.utils import first_token, fluency, next_token_logprobs

        prompt, correct, incorrect = record["prompt"], record["correct"], record["incorrect"]
        logprobs = next_token_logprobs(self.model, prompt)

        top_vals, top_idx = logprobs.topk(self.cfg.top_k)
        top_k = [(self.model.to_string(i.item()), round(v.item(), 3))
                 for v, i in zip(top_vals, top_idx)]

        correct_lp = logprobs[first_token(self.model, correct)].item()
        incorrect_lp = (logprobs[first_token(self.model, incorrect)].item()
                        if incorrect else None)
        logit_diff = (correct_lp - incorrect_lp) if incorrect else None

        generation = generate(self.model, prompt, self.cfg)
        continuation = generation[len(prompt):] if generation.startswith(prompt) else generation

        return {
            "prompt_id": record["id"],
            "suite": record["suite"],
            "split": record["split"],
            "prompt": prompt,
            "correct": correct,
            "incorrect": incorrect,
            "top1_token": top_k[0][0].strip(),
            "top1_logprob": round(top_k[0][1], 3),
            "top_k": json.dumps(top_k, ensure_ascii=False),
            "correct_logprob": round(correct_lp, 3),
            "incorrect_logprob": round(incorrect_lp, 3) if incorrect_lp is not None else None,
            "logit_diff": round(logit_diff, 3) if logit_diff is not None else None,
            "prefers_correct": bool(logit_diff > 0) if logit_diff is not None else None,
            "generation": generation,
            "continuation": continuation,
            "fluency": fluency(self.model, generation),
        }

    def run_behavioral_batch(self, records: list[dict]) -> pd.DataFrame:
        """Score every prompt in the inference batch; a failed prompt never kills the run."""
        rows = []
        for i, record in enumerate(records, start=1):
            sample_id = f"S{i:02d}"
            try:
                row = {"sample_id": sample_id, **self.score_prompt(record), "status": "ok"}
                self.log.info("%s [%s] %-46s -> %-12s logit_diff=%s", sample_id, row["suite"],
                              one_line(row["prompt"], 46), one_line(row["top1_token"], 12),
                              f"{row['logit_diff']:+.2f}" if row["logit_diff"] is not None
                              else "n/a")
            except Exception as exc:                        # keep the batch going
                self.log.warning("%s failed on %s: %s", sample_id, record.get("id"), exc)
                row = {"sample_id": sample_id, "prompt_id": record.get("id"),
                       "suite": record.get("suite"), "prompt": record.get("prompt"),
                       "status": f"error: {exc}"}
            rows.append(row)
        return pd.DataFrame(rows)

    # -------------------------------------------------------- edit-and-generate pass
    def run_edit_batch(self, targets: list[dict],
                       edit_layer: Optional[int] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Edit each factual target, generate under the edit, and score the result."""
        from models.metrics import evaluate_edit

        per_prompt, summaries = [], []
        for i, target in enumerate(targets, start=1):
            edit_id = f"E{i:02d}"
            self.log.info("%s editing '%s': %s -> %s", edit_id, target["subject"],
                          target["true"], target["new"])
            try:
                df, scores = evaluate_edit(self.model, target, self.cfg, layer=edit_layer)
            except Exception as exc:
                self.log.warning("%s failed: %s", edit_id, exc)
                summaries.append({"edit_id": edit_id, "subject": target["subject"],
                                  "status": f"error: {exc}"})
                continue

            df.insert(0, "edit_id", edit_id)
            df.insert(1, "subject", target["subject"])
            df["true"], df["new"] = target["true"], target["new"]
            per_prompt.append(df)
            summaries.append({"edit_id": edit_id, "subject": target["subject"],
                              "true": target["true"], "new": target["new"],
                              **scores, "status": "ok"})
            self.log.info("%s layer=%s efficacy=%.2f generalization=%.2f specificity=%.2f "
                          "(top-1 preserved %.2f) fluency %.2f -> %.2f",
                          edit_id, scores["layer"], scores["efficacy"],
                          scores["generalization"], scores["specificity"],
                          scores["specificity_pred_preserved"],
                          scores["fluency_before"], scores["fluency_after"])

        edit_df = pd.concat(per_prompt, ignore_index=True) if per_prompt else pd.DataFrame()
        return edit_df, pd.DataFrame(summaries)


# --------------------------------------------------------------------------- #
# Report writing
# --------------------------------------------------------------------------- #
def render_samples_text(behavioral: pd.DataFrame, edit_df: pd.DataFrame,
                        edit_summary: pd.DataFrame, meta: dict) -> str:
    """Build the human-readable outputs/samples.txt report."""
    lines = [banner("GENERATED SAMPLES — Opening the Black Box (Phase 2 pipeline)")]
    lines.append(f"model        : {meta['model']['name']} "
                 f"({meta['model']['n_params']:,} parameters, {meta['model']['n_layers']} layers)")
    lines.append(f"device       : {meta['model']['device']}")
    lines.append(f"decoding     : {'sampling' if meta['config']['gen_do_sample'] else 'greedy'}, "
                 f"max_new_tokens={meta['config']['gen_max_new_tokens']}, "
                 f"seed={meta['config']['seed']}")
    lines.append(f"generated at : {meta['run']['utc']} (UTC)")
    lines.append(f"command      : {meta['run']['command']}")
    lines.append("")

    lines.append(banner("PART A — behavioral prompts: next-token prediction + continuation",
                        char="-"))
    lines.append(
        "Each sample shows what the model predicts next, how strongly it prefers the correct\n"
        "answer over the matched incorrect one (logit difference; positive = correct), and a\n"
        "greedy continuation. These are the measurements Experiments 1-3 are built on.\n"
    )
    for _, r in behavioral.iterrows():
        if r.get("status") != "ok":
            lines.append(f"[{r['sample_id']}] {r.get('prompt_id')} — FAILED: {r.get('status')}\n")
            continue
        top_k = ", ".join(f"{tok!r} {lp:+.2f}" for tok, lp in json.loads(r["top_k"]))
        verdict = "correct" if r["prefers_correct"] else "INCORRECT"
        lines += [
            f"[{r['sample_id']}] suite={r['suite']}  split={r['split']}  id={r['prompt_id']}",
            f"  prompt        : {r['prompt']}",
            f"  correct/wrong : {r['correct']!r} ({r['correct_logprob']:+.2f})  vs  "
            f"{r['incorrect']!r} ({r['incorrect_logprob']:+.2f})",
            f"  logit diff    : {r['logit_diff']:+.3f}  -> model prefers the {verdict} answer",
            f"  top-{len(json.loads(r['top_k']))} next    : {top_k}",
            f"  continuation  : {one_line(r['continuation'], 240)}",
            f"  fluency       : {r['fluency']}  (mean 2/3-gram entropy of the generation)",
            "",
        ]

    if not edit_summary.empty:
        lines.append(banner("PART B — generative experiment: rank-one factual edit", char="-"))
        lines.append(
            "For each fact the pipeline localizes the storing MLP by causal tracing, fits a\n"
            "rank-one update to that layer's output matrix, and regenerates. 'before' and\n"
            "'after' differ only by that single edited weight matrix.\n"
        )
        for _, s in edit_summary.iterrows():
            if s.get("status") != "ok":
                lines.append(f"[{s['edit_id']}] {s['subject']} — FAILED: {s.get('status')}\n")
                continue
            lines.append(f"[{s['edit_id']}] {s['subject']}: {s['true']} -> {s['new']}   "
                         f"(edited MLP layer {int(s['layer'])})")
            rows = edit_df[edit_df.edit_id == s["edit_id"]]
            for kind, label in (("efficacy", "the edited prompt"),
                                ("generalization", "paraphrases (unseen wording)"),
                                ("specificity", "neighbourhood (must NOT change)")):
                subset = rows[rows.kind == kind]
                if subset.empty:
                    continue
                lines.append(f"  -- {kind}: {label}")
                for prompt in subset["prompt"].unique():
                    pair = subset[subset.prompt == prompt]
                    before = pair[pair.phase == "before"]["generation"]
                    after = pair[pair.phase == "after"]["generation"]
                    lines.append(f"     prompt : {prompt}")
                    if len(before):
                        lines.append(f"     before : {one_line(before.iloc[0], 200)}")
                    if len(after):
                        lines.append(f"     after  : {one_line(after.iloc[0], 200)}")
                    lines.append("")
            lines.append(
                f"  scores: efficacy={s['efficacy']:.2f}  generalization={s['generalization']:.2f}"
                f"  specificity={s['specificity']:.2f} "
                f"(top-1 unchanged on neighbours: {s['specificity_pred_preserved']:.2f})  "
                f"fluency {s['fluency_before']:.2f} -> {s['fluency_after']:.2f}\n"
            )

    lines.append(banner("END", char="-"))
    return "\n".join(lines) + "\n"


def build_run_metadata(cfg: ProjectConfig, model_facts: dict, manifest: dict,
                       behavioral: pd.DataFrame, edit_summary: pd.DataFrame,
                       files: dict, elapsed_s: float) -> dict:
    """Everything needed to interpret and reproduce this run, as JSON."""
    ok = behavioral[behavioral.status == "ok"] if "status" in behavioral else behavioral
    accuracy = (float(ok["prefers_correct"].mean()) if len(ok) and "prefers_correct" in ok
                else None)

    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("torch", "transformers", "transformer_lens", "pandas"):
        try:
            from importlib.metadata import version
            versions[package] = version(package)
        except Exception:                                   # package genuinely absent
            versions[package] = None

    edit_scores = {}
    if not edit_summary.empty and "efficacy" in edit_summary:
        done = edit_summary[edit_summary.status == "ok"]
        if len(done):
            edit_scores = {metric: round(float(done[metric].mean()), 3)
                           for metric in ("efficacy", "generalization", "specificity",
                                          "specificity_pred_preserved",
                                          "fluency_before", "fluency_after")
                           if metric in done}

    return {
        "run": {
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed_s, 1),
            "command": " ".join([os.path.basename(sys.executable), "src/model_runner.py"]
                                + sys.argv[1:]),
        },
        "model": model_facts,
        "config": {
            "seed": cfg.seed, "n_samples": cfg.n_samples, "top_k": cfg.top_k,
            "gen_max_new_tokens": cfg.gen_max_new_tokens, "gen_do_sample": cfg.gen_do_sample,
            "edit_lr": cfg.edit_lr, "edit_steps": cfg.edit_steps,
            "edit_kl_weight": cfg.edit_kl_weight, "edit_layer": cfg.edit_layer,
            "noise_scale": cfg.noise_scale,
        },
        "dataset": {
            "n_prompts_available": manifest["n_prompts"],
            "by_suite": manifest["by_suite"],
            "n_edit_targets_available": manifest["n_edit_targets"],
        },
        "results": {
            "n_behavioral_samples": int(len(behavioral)),
            "n_behavioral_ok": int(len(ok)),
            "next_token_accuracy": round(accuracy, 3) if accuracy is not None else None,
            "mean_logit_diff": (round(float(ok["logit_diff"].mean()), 3)
                                if len(ok) and "logit_diff" in ok else None),
            "n_edits": int(len(edit_summary)),
            "edit_scores_mean": edit_scores,
        },
        "versions": versions,
        "files": files,
    }


def write_figures(behavioral: pd.DataFrame, edit_summary: pd.DataFrame,
                  images_dir: str, model_name: str) -> dict:
    """Save the two summary figures; returns {name: repo-relative path}."""
    figures = {}
    ok = behavioral[behavioral.status == "ok"] if "status" in behavioral else behavioral
    if len(ok) and "logit_diff" in ok:
        by_suite = ok.groupby("suite")["logit_diff"].mean().to_dict()
        path = bar_chart(
            {k: round(float(v), 3) for k, v in by_suite.items()},
            os.path.join(images_dir, "logit_diff_by_suite.png"),
            title=f"Mean logit difference by behavior ({model_name})",
            ylabel="logit(correct) - logit(incorrect)", ylim=None)
        figures["logit_diff_by_suite"] = relative_to_repo(path)

    if not edit_summary.empty and "efficacy" in edit_summary:
        done = edit_summary[edit_summary.status == "ok"]
        if len(done):
            groups = {row["subject"]: {m: float(row[m])
                                       for m in ("efficacy", "generalization", "specificity")}
                      for _, row in done.iterrows()}
            path = grouped_bar_chart(
                groups, os.path.join(images_dir, "edit_quality.png"),
                title=f"Rank-one edit quality per target ({model_name})",
                ylabel="rate", ylim=(0, 1.05))
            figures["edit_quality"] = relative_to_repo(path)
    return figures


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run the end-to-end generation pipeline and save samples to outputs/.")
    ap.add_argument("--config", default=None, help="path to a YAML config file")
    ap.add_argument("--model", default=None, help="model name, e.g. pythia-160m | pythia-410m | gpt2")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"], help="force a device")
    ap.add_argument("--seed", type=int, default=None, help="random seed")
    ap.add_argument("--n-samples", type=int, default=None, dest="n_samples",
                    help="how many behavioral prompts to run (default 10)")
    ap.add_argument("--max-new-tokens", type=int, default=None, dest="gen_max_new_tokens",
                    help="tokens generated per prompt")
    ap.add_argument("--outputs-dir", default=None, dest="outputs_dir",
                    help="where to write samples (default outputs/)")
    ap.add_argument("--skip-edit", action="store_true",
                    help="skip the slower edit-and-generate experiment")
    ap.add_argument("--edit-layer", type=int, default=None,
                    help="force the edited MLP layer instead of running causal tracing")
    ap.add_argument("--n-targets", type=int, default=None,
                    help="limit how many edit targets are processed")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, model_name=args.model, device=args.device, seed=args.seed,
                      n_samples=args.n_samples, gen_max_new_tokens=args.gen_max_new_tokens,
                      outputs_dir=args.outputs_dir)

    outputs_dir = ensure_dir(cfg.abs_path(cfg.outputs_dir))
    images_dir = ensure_dir(os.path.join(outputs_dir, IMAGES_SUBDIR))
    log = get_logger("model_runner", log_file=os.path.join(outputs_dir, "run.log"))
    started = time.time()

    log.info("=== stage 1/4: dataset ===")
    set_seed(cfg.seed)
    manifest = load_manifest(cfg, logger=log)
    records = load_processed_prompts(cfg, logger=log)
    batch = select_inference_samples(records, n_samples=cfg.n_samples,
                                     per_suite=cfg.samples_per_suite, seed=cfg.seed)
    log.info("%d prompts available; selected %d for inference (%s)", len(records), len(batch),
             ", ".join(sorted({r["suite"] for r in batch})))

    runner = PipelineRunner(cfg, log)

    log.info("=== stage 2/4: model ===")
    facts = runner.model_facts()
    log.info("%s | %d params | %d layers x %d heads | d_model=%d", facts["name"],
             facts["n_params"], facts["n_layers"], facts["n_heads"], facts["d_model"])

    log.info("=== stage 3/4: inference on %d behavioral prompts ===", len(batch))
    behavioral = runner.run_behavioral_batch(batch)

    edit_df, edit_summary = pd.DataFrame(), pd.DataFrame()
    if args.skip_edit:
        log.info("=== stage 4/4: edit-and-generate SKIPPED (--skip-edit) ===")
    else:
        targets = load_processed_edit_targets(cfg, logger=log)
        if args.n_targets:
            targets = targets[: args.n_targets]
        log.info("=== stage 4/4: edit-and-generate on %d factual targets ===", len(targets))
        edit_df, edit_summary = runner.run_edit_batch(targets, edit_layer=args.edit_layer)

    # ---- write everything out
    files = {}
    behavioral.to_csv(os.path.join(outputs_dir, BEHAVIORAL_CSV), index=False)
    files["behavioral_csv"] = relative_to_repo(os.path.join(outputs_dir, BEHAVIORAL_CSV))
    if not edit_df.empty:
        edit_df.to_csv(os.path.join(outputs_dir, EDIT_CSV), index=False)
        files["edit_csv"] = relative_to_repo(os.path.join(outputs_dir, EDIT_CSV))
        edit_summary.to_csv(os.path.join(outputs_dir, EDIT_SUMMARY_CSV), index=False)
        files["edit_summary_csv"] = relative_to_repo(os.path.join(outputs_dir,
                                                                  EDIT_SUMMARY_CSV))
    files.update(write_figures(behavioral, edit_summary, images_dir, cfg.model_name))

    meta = build_run_metadata(cfg, facts, manifest, behavioral, edit_summary, files,
                              time.time() - started)
    samples_path = write_text(os.path.join(outputs_dir, SAMPLES_TXT),
                              render_samples_text(behavioral, edit_df, edit_summary, meta))
    meta["files"]["samples_txt"] = relative_to_repo(samples_path)
    write_json(os.path.join(outputs_dir, RUN_METADATA_JSON), meta)

    results = meta["results"]
    log.info("done in %.1fs | %d samples | next-token accuracy %s | mean logit diff %s",
             meta["run"]["elapsed_seconds"], results["n_behavioral_samples"],
             results["next_token_accuracy"], results["mean_logit_diff"])
    if results["edit_scores_mean"]:
        log.info("edit means: %s", results["edit_scores_mean"])
    log.info("outputs -> %s", relative_to_repo(outputs_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
