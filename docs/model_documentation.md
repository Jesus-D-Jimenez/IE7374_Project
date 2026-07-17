# Model Documentation

*Component 2 — Model choices, architecture, training/editing procedures, and evaluation*

## 1. Framework selection

| Choice | Why |
|---|---|
| **PyTorch** | The backend both target models are distributed in; needed for the gradient-based `v*` optimization. |
| **Hugging Face Transformers** | Canonical source of the pretrained Pythia / GPT-2 weights and tokenizers. |
| **TransformerLens** | The decisive choice — it exposes named hooks into *every* internal activation (attention `z`, MLP `post`, `resid_post`, …) and supports ablation and activation patching. Interpretability and editing are impractical without this level of access. |
| **NumPy / pandas / matplotlib** | Analysis and figures. |

## 2. Model architecture (object of study)

We use pretrained **decoder-only autoregressive transformers** (GPT-NeoX architecture for Pythia,
GPT-2 architecture for the control), loaded through TransformerLens's `HookedTransformer`. Each
block is causal self-attention + an MLP with a residual stream:

```
resid -> LN -> Attention (n_heads) -> +resid -> LN -> MLP (W_in, act, W_out) -> +resid
```

Key components the project acts on:

- `blocks[l].attn.hook_z` — per-head attention output (ablated in Experiment 3a).
- `blocks[l].mlp.hook_post` — MLP hidden activation; the **key** `k` for editing.
- `blocks[l].hook_mlp_out` — MLP output; the **value** `v` we edit.
- `blocks[l].hook_resid_post` — residual stream (patched during causal tracing).

No architecture is trained. See [`../models/loader.py`](../models/loader.py).

## 3. "Training" = fitting the edit (Experiment 4)

There is no model training or fine-tuning. The only optimization is the **ROME-style edit**, which
has two steps ([`../models/editing.py`](../models/editing.py)):

**(a) Optimize the target value `v*`.** At the subject's last token in the chosen layer `L`, we add
a trainable vector `δ` to `hook_mlp_out` and minimize

```
loss = −log p(new_object | prompt)  +  λ · ||δ||²
```

with Adam (`edit_lr`, `edit_steps`, `edit_kl_weight` in [`../config/default.yaml`](../config/default.yaml)).
Then `v* = v_orig + δ`.

**(b) Solve the rank-one weight update.** With key `k = mlp.hook_post[L, pos]`, we set

```
W_out' = W_out + (k / (k·k)) ⊗ (v* − v_orig)
```

so the layer maps `k → v*` while perturbing other directions minimally. Original weights are cached
so every edit is reversible.

> **Simplification vs. full ROME.** Meng et al. whiten `k` by a second-moment matrix `C` estimated
> from a corpus. We start with `C = I` (identity), which keeps the method self-contained. The
> specificity metric is precisely what exposes the cost of this simplification; replacing `C` with
> Pile-estimated statistics is the documented improvement path (and MEMIT the multi-fact extension).

The **edit layer** is chosen by **causal tracing** (Experiment 3,
[`../models/interpret.py`](../models/interpret.py)): corrupt the subject with ~3σ embedding noise,
patch the clean residual stream back one (layer, position) at a time, and pick the layer with
maximum recovery of the true-object probability at the subject's last token.

## 4. Evaluation and metrics

Implemented in [`../models/metrics.py`](../models/metrics.py).

**Interpretability (Experiments 1–3)**
- **Logit difference**: `logit(correct) − logit(incorrect)` for the next token.
- **Ablation drop**: fall in logit difference when a head/MLP is zeroed.
- **Selectivity**: normalized difference between the drop on targeted vs. control prompts.

**Generation (Experiment 4)** — Meng et al. (2022) framework, read off *generated* text:
- **Efficacy** — the edited prompt now generates the new object.
- **Generalization** — paraphrase prompts also generate the new object.
- **Specificity** — neighborhood (unrelated) facts keep their original answers.
- **Fluency** — bi-/tri-gram entropy of the generation stays high (no degeneration).

## 5. Reproducibility

- Single YAML config with CLI overrides; fixed seeds; greedy decoding for stable before/after
  comparisons.
- Deterministic dataset generation (`data/build_suites.py`).
- Headless scripts (`scripts/`) mirror the notebooks (`experiments/`) so results can be regenerated
  without a GUI.
