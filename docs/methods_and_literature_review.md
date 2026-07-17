# Methods and Literature Review

*Component 1 — Research and Selection of Methods*

## 1. Objectives

The project is a **mechanistic interpretability** study of a small generative language model,
extended with a **generative editing** step so that the work does not only *analyze* the model
but also *produces and evaluates* model outputs. Concretely, the tasks are:

1. **Component identification** — for three behaviors (subject–verb agreement, factual recall,
   induction), find the individual attention heads and MLP layers that causally drive them.
2. **Selective ablation** — confirm those components are the real mechanism by switching them off
   and checking the effect is specific to the targeted behavior.
3. **Editing and generation** — use the localized factual-recall components to apply a rank-one
   weight edit, then have the edited model **generate text** and score it.

These map onto course units on attention, transformers, and pre-training: each question asks what
the transformer stack is actually computing, one component at a time.

## 2. Literature Review

| Work | Contribution we use |
|---|---|
| Vaswani et al. (2017), *Attention Is All You Need* | The transformer / self-attention architecture under study. |
| Radford et al. (2019), GPT-2 | Decoder-only autoregressive LM; our comparison model and the field's most-studied target. |
| nostalgebraist (2020), *logit lens* | Decoding intermediate residual streams to watch a prediction form across depth (Experiment 2). |
| Elhage et al. (2021); Olsson et al. (2022), *Induction Heads* | Identify induction heads that copy repeated patterns; our induction suite reproduces this. |
| Wang et al. (2022), *Interpretability in the Wild (IOI)* | Circuit-level analysis via activation patching / path patching; the template for Experiment 3. |
| Meng et al. (2022), *ROME — Locating and Editing Factual Associations in GPT* | Causal tracing to localize a fact in an MLP, then a **rank-one edit**; the basis of Experiment 4 and its evaluation (efficacy / generalization / specificity / fluency). |
| Meng et al. (2023), *MEMIT* | Scales editing to many facts; noted as the extension path beyond a single rank-one edit. |
| Biderman et al. (2023), *Pythia* | The model suite we use — public training data (the Pile) and 154 checkpoints, purpose-built for interpretability. |
| Nanda & Bloom (2022), *TransformerLens* | The tooling: hooks into every internal activation, ablation, and patching. |

**Gap this project sits in.** Most published mechanistic work targets GPT-2. Pythia additionally
ships its training data and intermediate checkpoints, which lets us (as a stretch goal) ask *when*
during training a mechanism appears — something closed models do not allow.

## 3. Method Selection

- **Observation vs. intervention.** Attention maps and the logit lens are observational and cheap,
  but correlational. Causal methods — **ablation** and **activation patching / causal tracing** —
  are what license causal claims, so they are the backbone of Experiment 3.
- **Editing method.** We adopt **ROME** (Meng et al. 2022) because it is (a) already causal-tracing
  based, so it reuses Experiment 3's localization, and (b) comes with a standard four-part
  evaluation of *generated* outputs. See [`model_documentation.md`](model_documentation.md) for the
  exact update rule and the identity-covariance simplification we start from.
- **Why not fine-tuning / full training.** The research questions are about mechanisms already
  present in a pretrained model. Fine-tuning would change the object of study and add compute for
  no scientific gain. The only weight change we make is the targeted rank-one edit.

## 4. Preliminary Experiments (feasibility)

Before committing, we validated feasibility with small-scale checks, documented alongside the code:

- **Model size / compute.** `pythia-160m` and `gpt2` both load and run on CPU inside a single
  notebook; a full head-ablation scan over one prompt is `n_layers × n_heads` forward passes
  (144 for pythia-160m) and completes in well under a minute.
- **Signal exists.** The baseline logit difference is positive on known facts (e.g. *"The capital
  of France is" → "Paris"*), confirming the behaviors are present to be dissected
  (`tests/test_smoke.py::test_logit_diff_prefers_correct`).
- **Editing is tractable.** The rank-one edit is a closed-form update to one weight matrix plus a
  short (~25-step) optimization of the target vector — no retraining — so Experiment 4 fits the
  timeline.

Adjustments made from these checks: fixed greedy decoding for reproducible before/after
generations; single-token answer verification in preprocessing so probability comparisons are
clean; and starting from the identity-covariance ROME simplification, with corpus second-moment
statistics flagged as the improvement path.
