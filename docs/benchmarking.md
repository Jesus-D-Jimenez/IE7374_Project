# Benchmarking and Model Comparison

*Component 1 — Benchmarking*

We compared candidate models along the axes that matter for a mechanistic-interpretability +
editing project: whether the internals are analyzable, whether pretrained weights are available,
computational cost, and how well the tooling supports them.

## Candidate models

| Model | Params | Pretrained | Training data public | Checkpoints | TransformerLens support | Runs on CPU |
|---|---|---|---|---|---|---|
| **Pythia-160m** (main) | 160M | ✅ | ✅ (the Pile) | ✅ 154 | ✅ | ✅ |
| **Pythia-410m** (main, larger) | 410M | ✅ | ✅ (the Pile) | ✅ 154 | ✅ | ✅ (slower) |
| **GPT-2 small** (comparison) | 124M | ✅ | ❌ | ❌ | ✅ | ✅ |
| GPT-2 medium/large | 355M–774M | ✅ | ❌ | ❌ | ✅ | ⚠️ slower |
| Llama-2 7B | 7B | ✅ | ❌ | ❌ | partial | ❌ (needs GPU) |

## Decision factors

- **Analyzability over raw accuracy.** For interpretability the figure of merit is not benchmark
  accuracy but whether we can *read and intervene on* the internals. Small decoder-only models with
  clean, documented architectures win.
- **Reproducibility.** Pythia's public training corpus and 154 checkpoints are unique among these
  options and enable the stretch question (when a mechanism emerges during training).
- **Compute / scalability.** 160M runs comfortably on CPU; 410M is the same architecture at larger
  scale, letting us test whether findings hold as models grow without changing tooling.
- **Pretrained availability + tooling.** All chosen models are on Hugging Face and fully supported
  by TransformerLens, which provides the activation hooks the whole method depends on.

**Choice:** `pythia-160m` and `pythia-410m` as the primary models, with `gpt2` as a replication
control (Experiment 5) because it is the most-studied model in the literature and lets us verify
our methods reproduce published induction-head and editing results.

## Metrics used to compare *runs* (not just models)

Because we do not train a classifier, "performance" is measured with task-appropriate quantities:

- **Logit difference** — margin between correct and incorrect next token (behavioral signal).
- **Ablation drop / selectivity** — causal contribution of a component and how specific it is.
- **Edit efficacy / generalization / specificity / fluency** — quality of *generated* text after a
  ROME-style edit, following Meng et al. (2022).

See [`model_documentation.md`](model_documentation.md) for definitions and
[`../results`](../results) for the generated tables and figures.
