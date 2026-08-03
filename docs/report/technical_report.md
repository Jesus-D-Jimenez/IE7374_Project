---
title: "Opening the Black Box: Locating, Ablating, and Editing the Mechanisms of a Small Generative Language Model"
author: "Jesus D. Jimenez Ballestas — Group 25 · IE7374 Generative AI · Northeastern University"
date: "August 2026"
---

# 1. Introduction and Motivation

A generative language model is usually treated as a single opaque function: text goes in, text
comes out, and the only handle on its behavior is the prompt. That framing is expensive — when
such a model states a wrong fact there is nothing to fix short of retraining or fine-tuning, both
of which change the whole network to correct one thing.

This project takes the opposite approach on a model small enough to study exhaustively. It treats
EleutherAI's **Pythia-160m** as a system with parts, locates the parts responsible for three
behaviors, and uses that localization to **change what the model generates** by editing a single
weight matrix. Nothing is trained or fine-tuned: the pretrained weights are used exactly as
released, and the only weight change anywhere is one deliberate rank-one update. Localization is
the precondition for what people actually want from interpretability — correcting outdated facts,
auditing what a model memorized, predicting how it fails — and editing is the cheap alternative to
retraining if one fact can be rewritten without disturbing the rest.

**Research questions.** The project follows the four questions set out in the proposal:

* **RQ1.** Which attention heads and MLP layers causally drive subject–verb agreement, factual
  recall, and induction?
* **RQ2.** When those components are ablated, does performance drop *selectively* on the targeted
  behavior, or does everything degrade at once?
* **RQ3 (generative core).** Can the localized components be used to apply a rank-one edit such
  that the model *generates* text consistent with the new fact — and does that generated text
  satisfy efficacy, generalization, specificity, and fluency?
* **RQ4.** Using Pythia's 154 training checkpoints, when during pretraining does each behavior
  appear, and does it emerge gradually or abruptly?

**Contributions.** Beyond the planned pipeline, four results came from testing an assumption
rather than accepting it:

1. The edit layer matters far less than assumed — a fact can be rewritten from **most** early
   layers — and the multi-layer tracing window proposed as the fix makes layer selection *worse*
   (§5.3).
2. Subject–verb agreement is carried by a strikingly **selective** head (L6.H4, selectivity 1.00),
   while factual recall and induction are diffuse and MLP-dominated (§5.4).
3. The three behaviors emerge at **different times and in different shapes** during pretraining,
   and the final checkpoint is *not* the best one for two of them (§5.5).
4. The edit method's apparent failure on GPT-2 is a **hyperparameter–scale mismatch**, not a
   property of GPT-2: the objective penalizes the injected vector in absolute terms while the
   vector's required size scales with the residual-stream norm, which differs ~6× between these
   models (§5.7). Fixing the penalty turns efficacy 0.00 into 1.00 and exposes a
   generalization/specificity trade-off the original setting had hidden.

# 2. Related Work

**Transformer internals.** The architecture under study is the decoder-only transformer of
Vaswani et al. [1] as popularized by GPT-2 [2]; the framing used here — the residual stream as a
channel that attention heads and MLPs write into — follows Elhage et al. [3].

**Observation.** The *logit lens* (nostalgebraist [4]) decodes intermediate residual streams
through the unembedding to watch a prediction take shape across depth. Being purely observational
is its limitation: it shows correlation between a layer's state and the final answer, not
causation.

**Intervention.** Olsson et al. [5] identified *induction heads* that copy repeated patterns and
linked their formation to an abrupt phase change in training loss. Wang et al. [6] traced the
indirect-object-identification circuit in GPT-2 with activation and path patching, establishing
the ablate-and-measure methodology applied in §5.4.

**Localization and editing.** Meng et al.'s ROME [7] is the direct ancestor of this project's
generative half: corrupt the subject tokens, restore clean activations layer by layer to find
where the fact is stored (causal tracing), then apply a rank-one update to that MLP's output
matrix so a key vector maps to a new value. ROME also supplies the four-part evaluation of
*generated* text used throughout §5; MEMIT [8] scales the idea to thousands of facts. The
implementation here starts from ROME's identity-covariance simplification (§4.3). Pythia [9] is
the primary model rather than GPT-2 because it ships 154 intermediate checkpoints and a public
training corpus, which is what makes RQ4 answerable; internal access is through TransformerLens
[10].

**Gap.** Published mechanistic work overwhelmingly targets GPT-2 at a single point in training,
and published editing work reports aggregate scores at hyperparameters tuned for one model. This
project sits in the intersection usually skipped: the same three behaviors measured *across two
model families, three scales, and eleven training checkpoints*, with the editing hyperparameters
themselves treated as an object of study rather than a constant.

# 3. Dataset

The project uses hand-authored prompt data rather than a downloaded corpus, because the
measurement — a difference in probability between two carefully matched continuations — requires
control the public benchmarks do not give.

## 3.1 Behavioral suites

**48 prompts across three behaviors**, each a *minimal pair*: a prompt, a correct next token, and
a matched incorrect one that differs in exactly the property being tested.

| Suite | Prompts | Example prompt | Correct | Incorrect |
|---|---:|---|---|---|
| Subject–verb agreement | 20 | `The keys on the table` | ` are` | ` is` |
| Factual recall | 12 | `The capital of France is` | ` Paris` | ` London` |
| Induction | 16 | `AliceBob Alice` | `Bob` | `Carol` |

: Table 1. The three behavioral suites. Agreement and induction prompts are stored as contrast
pairs whose correct answers are swapped, so a model cannot score well by ignoring the prompt.

Agreement prompts include plural attractors (`The key on the table` vs. `The keys on the table`),
catching a model that relies on the nearest noun rather than the grammatical subject. Induction
prompts are synthetic `ABAB` patterns whose continuation cannot be memorized — recoverable only by
copying from earlier in the same prompt, the induction-head behavior of Olsson et al. [5].

## 3.2 Edit targets

**3 factual targets** — *The Eiffel Tower* (Paris → Rome), *Mount Everest* (Nepal → Canada), and
*The Mona Lisa* (Leonardo → Rembrandt) — each carrying the full ROME evaluation structure: the
fact to rewrite, two paraphrases asking it in unseen wording (*"You can find the Eiffel Tower in
the heart of"*), and two neighbourhood facts that must *not* change (*"Big Ben is located in the
city of"*). Each target therefore contributes 5 evaluation prompts, generated before and after the
edit — 30 generations per model per configuration.

## 3.3 Preprocessing

`src/data_loader.py` normalizes whitespace, verifies that **every answer is a single BPE token**
(multi-token answers make the log-probability comparison ill-defined), checks that each edit
target's subject occurs in its prompt, and splits the prompts 70/15/15 with the constraint that
**a contrast pair never straddles a split**. The processed dataset ships with a manifest of
SHA-256 hashes and is committed; CI rebuilds it on every push to confirm the build is
byte-for-byte deterministic.

**Scope note.** The proposal projected 300–500 prompts and 20–40 edit targets; the delivered
dataset is 48 and 3 — hand-verified, but the single largest limit on the statistical weight of the
results (§6.4).

# 4. Methodology

## 4.1 Measurement

Every behavioral claim reduces to one quantity, the **logit difference**

$$\Delta = \text{logit}(\text{correct}) - \text{logit}(\text{incorrect})$$

evaluated at the final position. Because both answers are single tokens drawn from the same
softmax, the normalizer cancels and $\Delta$ is identical whether computed on logits or
log-probabilities — so a single forward pass yields the margin, the top-$k$ predictions, and the
model's own answer at once. Positive $\Delta$ means the model prefers the correct continuation.

## 4.2 Localization (RQ1, RQ2)

Three techniques, in increasing order of causal strength:

* **Logit lens** (observational): decode each layer's residual stream at the final position
  through the unembedding, giving a per-layer trajectory of how the prediction forms.
* **Zero-ablation** (interventional): set one attention head's output ($z$) or one MLP's output to
  zero and re-measure $\Delta$; the drop is that component's causal contribution. A full head scan
  is $n_{\text{layers}} \times n_{\text{heads}} = 144$ forward passes for pythia-160m.
* **Causal tracing** (interventional, ROME-style): corrupt the subject's token embeddings with
  Gaussian noise at 3× the embedding standard deviation, then restore the clean residual stream one
  (layer, position) at a time and measure how much of the answer's log-probability returns. The
  layer with peak recovery at the subject's last token is where the fact is read.

**Selectivity** turns a drop into evidence of a *mechanism*:
$(d_{\text{target}} - d_{\text{control}}) / (|d_{\text{target}}| + |d_{\text{control}}|)$, where
$d_{\text{control}}$ is the same head's mean ablation drop on the *other* two behaviors. 1.0 means
the head matters only for its own behavior; 0.0 means it is generic machinery.

## 4.3 Editing and generation (RQ3)

Given the traced layer $\ell$, the edit proceeds in two steps.

**Step 1 — find the target value.** Optimize a delta $\delta$ added to the layer's MLP output at
the subject's last token so the model predicts the new object $o^*$ — the only optimization in the
project, 25 Adam steps at learning rate 0.5 with $\lambda = 0.0625$ (`edit_kl_weight`):

$$\mathcal{L}(\delta) = -\log p(o^* \mid \text{prompt}) + \lambda \lVert \delta \rVert^2$$

**Step 2 — write it into the weights.** With $k$ the layer's post-activation key vector at that
position and $v^* = v_{\text{orig}} + \delta$, apply the rank-one update

$$W_{\text{out}} \mathrel{+}= \frac{k}{k^\top k} (v^* - v_{\text{orig}})^\top$$

so that $k \mapsto v^*$ while the mapping stays rank-one. This is ROME's update with the
corpus-estimated second-moment matrix $C$ replaced by the identity, which keeps the method
self-contained. Edits are applied through a context manager that restores the original weights
afterwards, so a failed edit cannot contaminate the next measurement.

**Evaluation of generated text** follows Meng et al. [7]: **efficacy** (does the edited prompt's
generation contain the new object?), **generalization** (do the paraphrases?), **specificity** (do
neighbourhood generations still contain their own answers?), and **fluency** (mean 2- and 3-gram
entropy; lower means more repetitive). Plain specificity scores 0 whenever the base model never
knew the neighbourhood fact, conflating "the edit broke it" with "the model never had it", so a
stricter companion is reported throughout: **`specificity_pred_preserved`**, the share of
neighbourhood prompts whose top-1 next token is *unchanged* by the edit.

## 4.4 Reproducibility

Decoding is greedy, every random draw is seeded through one helper, and all hyperparameters live
in one YAML file loaded into a typed dataclass. The pipeline runs with one command
(`python src/model_runner.py`, ~60 s on CPU) and writes generations, tables, figures, and a
`run_metadata.json` recording package versions and every configuration value. Re-running
reproduces the committed numbers exactly; a Windows-vs-Linux-container check reproduced all 10
generations and top-1 predictions identically, with log-probabilities agreeing to ~1e-3. A
38-test pytest suite runs in CI, including checks that the committed dataset and result tables are
complete and well-formed.

# 5. Experiments and Results

Ten experiments were run. Experiments 1–4 are the pipeline proposed at the outset (baseline,
logit lens, ablation/tracing, edit-and-generate); 5–10 were added for this milestone, each because
a result from 1–4 rested on an assumption worth testing. Every number below is reproduced by the
committed artifacts in `outputs/` and `outputs/study/`.

## 5.1 Baseline behavior (Experiment 1)

On pythia-160m the model prefers the correct continuation on **43 of 48 prompts**
(mean $\Delta = +3.02$): agreement 20/20 ($+4.62$), factual recall 10/12 ($+2.07$), induction
13/16 ($+1.73$). Agreement survives plural attractors, so the model tracks the grammatical subject
rather than the nearest noun. Factual recall is correct but weakly held — for *"The capital of
Spain is"* the model ranks ` Madrid` behind generic continuations: it *knows* the fact in the
margin sense while not *saying* it.

## 5.2 Where the answer forms (Experiment 2)

![Figure 1. Logit lens by behavior: agreement resolves by mid-stack, factual recall much later, and every margin falls at the last layer.](outputs/study/logit_lens.png){width=4.5in}

The agreement margin reaches $+8.1$ by layer 6 and peaks at $+12.3$ at layer 9; factual recall is
still near zero at layer 6 ($+0.33$) and only separates at layers 8–9 — consistent with agreement
being an early syntactic computation while factual recall requires information moved from the
subject tokens later in the stack, the same asymmetry §5.4 finds independently.

All three margins then *drop* sharply at layer 11 (agreement $+12.3 \rightarrow +4.5$): the final
block reshapes the representation for the unembedding rather than sharpening the decision, so
reading the logit lens at the last layer alone would understate every behavior.

## 5.3 Which layer can hold an edit (Experiment 6)

One target (*The Eiffel Tower*) had been failing at its traced layer. Rather than assume the
tracing rule was at fault, the same edit was applied at **every** layer — 36 rank-one fits.

![Figure 2. Edit quality at every layer. Shaded bands are the layers causal tracing selects across five seeds; the last three layers never work.](outputs/study/layer_sweep.png){width=4.5in}

| Target | Layers with efficacy 1.0 | Fraction |
|---|---|---:|
| Mount Everest | 0–8 | 9/12 |
| The Eiffel Tower | 0–5, 7, 8 | 8/12 |
| The Mona Lisa | 2, 4, 5, 6, 7 | 5/12 |

: Table 2. Layers at which each fact can be successfully rewritten (pythia-160m).

Two findings follow. The failure at layer 6 for *The Eiffel Tower* is an **isolated dead layer**,
not a depth effect — its neighbours 5 and 7 both work. And editing never works in the last three
layers, which fits §5.2: by layer 9 the answer is already decided, so an edit there arrives too
late to change what the model generates.

Scoring each *layer-selection rule* by the swept quality of the layer it picks (5 tracing seeds ×
3 targets) settles the question the earlier milestone left open:

| Rule | Mean efficacy | Mean generalization | Mean top-1 preserved |
|---|---:|---:|---:|
| raw argmax (`window=1`, current default) | **0.933** | 0.033 | 0.800 |
| windowed argmax (`window=3`) | 0.733 | 0.167 | 0.500 |
| windowed argmax (`window=5`) | 0.800 | 0.167 | 0.667 |

: Table 3. The multi-layer tracing window — planned as a fix — selects worse layers than the raw
argmax it was meant to replace.

The raw rule lands on a working layer in 14 of 15 (seed, target) combinations. The windowed rule
was implemented anyway (`models/interpret.py::best_edit_layer`), but the measurement is why it is
not the default.

## 5.4 What carries each behavior (Experiments 3, 7)

Zero-ablating every attention head and every MLP, averaged over four prompts per behavior:

![Figure 3. Head-ablation drop per behavior, shared colour scale. Agreement concentrates in L6.H4; the other behaviors are spread across many heads.](outputs/study/ablation_heatmaps.png){width=4.5in}

| Behavior | Top head | Drop on own behavior | Mean drop on the others | Selectivity |
|---|---|---:|---:|---:|
| Agreement | **L6.H4** | +2.29 | −0.02 | **1.00** |
| Induction | L0.H0 | +1.19 | +0.17 | 0.75 |
| Factual recall | L0.H7 | +1.89 | +0.60 | 0.52 |

: Table 4. Selectivity of each behavior's most important head. 1.00 means the head matters only
for its own behavior.

L6.H4 answers RQ2 in the strongest form available at this scale: switching it off costs 2.29
logits of agreement margin and *nothing measurable* elsewhere — a dedicated component, not a
shared pathway. Factual recall's top head scores only 0.52 because it is also worth +0.94 to
agreement, the signature of a diffuse mechanism.

The largest single component for *every* behavior, however, is **MLP 0** (drops of +4.96, +3.59,
+2.69): shared input processing rather than behavior-specific circuitry. "Largest ablation drop"
and "the mechanism" are not the same claim, which is why selectivity is the metric that matters.

## 5.5 When the behaviors appear (Experiment 8, RQ4)

Pythia's 154 intermediate checkpoints turn "which components carry this behavior" into "when did
they get there". Eleven checkpoints were scored on all 48 prompts.

![Figure 4. Behavior emergence during pretraining: agreement switches on abruptly, induction climbs steadily, factual recall is wrong before it is right.](outputs/study/checkpoint_emergence.png){width=4.5in}

| Step | 0 | 512 | 1 000 | 2 000 | 4 000 | 16 000 | 64 000 | 143 000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Agreement | +0.02 | +0.15 | **+2.03** | +3.53 | +4.28 | +4.60 | +5.11 | +4.62 |
| Factual recall | +0.02 | −0.70 | +0.08 | +0.42 | +0.86 | +2.44 | **+2.90** | +2.07 |
| Induction | −0.02 | +0.08 | +0.25 | +0.84 | +1.13 | +1.41 | **+1.96** | +1.73 |

: Table 5. Mean logit difference by pretraining step (abbreviated; full table in
`outputs/study/checkpoint_summary.csv`).

Three observations. **(i)** Agreement appears abruptly — a factor-of-13 jump between steps 512 and
1 000 — and is complete by step 4 000, while induction rises gradually over an order of magnitude
of training. **(ii)** Factual recall is *worse than chance* at steps 128–512 ($-0.40$, $-0.70$):
the model has learned that "The capital of X is" is followed by a city, and picks the wrong one,
before it has learned which — a behavior getting worse on its way to getting better, which a
single end-of-training measurement cannot see. **(iii)** For two of three behaviors the **final
checkpoint is not the best**: step 64 000 beats step 143 000 on factual recall (+2.90 vs. +2.07)
and induction (+1.96 vs. +1.73).

## 5.6 The generative core: edit and generate (Experiments 4, 9, RQ3)

At the default configuration the edited model does generate text consistent with the rewritten
fact. A representative pair, pythia-160m, *Mount Everest: Nepal → Canada*:

> **before** — *Mount Everest is located in the country of Nepal. The mountain is located in the
> Himalayas, and is the highest mountain in the world…*
>
> **after** — *Mount Everest is located in the country of Canada. The Mount Everest National Park
> is located in the country of Canada. The park is located in the…*

Only one weight matrix differs between those generations. The identical evaluation on all three
models:

| Model | Efficacy | Generalization | Specificity | Top-1 preserved | Fluency (before → after) |
|---|---:|---:|---:|---:|---|
| gpt2 | 0.00 | 0.00 | 0.33 | 1.00 | 4.81 → 4.74 |
| pythia-160m | 0.67 | 0.00 | 0.33 | 0.83 | 4.42 → 4.52 |
| pythia-410m | **1.00** | **0.50** | 0.50 | 1.00 | 4.80 → 4.77 |

: Table 6. The same edit in three models, default hyperparameters. Fluency is essentially
unaffected everywhere — the edit changes what the model says, not whether it can say it.

Read on its own, Table 6 supports a tidy conclusion: rank-one editing works better at scale and
does not work on GPT-2 at all. Experiment 10 shows that conclusion is wrong.

## 5.7 Why the hyperparameters did not transfer (Experiment 10)

The edit objective penalizes the injected vector in absolute terms,
$\mathcal{L} = -\log p(o^*) + \lambda \lVert \delta \rVert^2$, but the size of a vector that
changes the prediction depends on the norm of the residual stream it is added to — and those norms
are not comparable across these models. At the edited layer the mean residual norm is **75.0 for
GPT-2** against **13.4** (pythia-160m) and **18.9** (pythia-410m), reaching 444 in GPT-2's last
layer. The same $\lambda$ is therefore a far harsher constraint on GPT-2: the optimizer drives
$\lVert\delta\rVert$ to 2.76 — where Pythia reaches 6–7 — with the loss *increasing* over its 25
steps. GPT-2's edit was never attempted; it was regularized out of existence.

![Figure 5. Left: what the penalty trades away (3 models x 3 targets). Right: the delta size each penalty permits; at the default GPT-2's collapses to 2.76.](outputs/study/edit_penalty.png){width=4.5in}

| `edit_kl_weight` | Mean ‖δ‖ | Efficacy | Generalization | Top-1 preserved |
|---|---:|---:|---:|---:|
| 0.0625 (default) | 5.1 | 0.556 | 0.167 | **0.944** |
| 0.01 | 12.4 | **0.889** | 0.444 | 0.722 |
| 0.001 | 29.6 | **0.889** | 0.500 | 0.389 |
| 0 | 91.3 | 0.778 | **0.611** | 0.000 |

: Table 7. Edit-penalty sweep pooled over 3 models × 3 targets. The default is the
high-specificity corner of a trade-off, not a neutral choice.

Two conclusions follow, and they are the most important results in this report. **The GPT-2
"failure" is a hyperparameter artifact** — at $\lambda = 0.001$ GPT-2 reaches efficacy 1.00, at
$\lambda = 0.01$, 0.67. And **"generalization = 0.00" was also an artifact**: the earlier
milestone attributed it to the identity-covariance simplification, but pooled generalization rises
from 0.167 to 0.611 as the penalty falls while top-1 preservation collapses from 0.944 to 0.000.
The configuration sat at one end of a trade-off a single run could not reveal; at
$\lambda = 0.01$ pythia-410m reaches efficacy 1.00, generalization 0.83, and top-1 preserved 0.83
simultaneously. The committed default was deliberately **not** re-tuned to the better-looking
setting, so the pipeline's headline output remains the conservative configuration of earlier
milestones and the trade-off is documented rather than hidden.

## 5.8 Behavior across models and scales (Experiment 5)

| Model | Agreement | Factual recall | Induction | All 48 |
|---|---|---|---|---|
| gpt2 (124M) | +4.50 (1.00) | +2.79 (0.92) | +2.45 (0.94) | +3.39 (0.96) |
| pythia-160m (162M) | +4.62 (1.00) | +2.07 (0.83) | +1.73 (0.81) | +3.02 (0.90) |
| pythia-410m (405M) | +5.12 (1.00) | **+4.41 (1.00)** | +2.42 (0.88) | +4.04 (0.96) |

: Table 8. Mean logit difference (accuracy in parentheses) across models. Agreement is saturated
at every scale; factual recall is the behavior that scales.

Agreement is at ceiling in all three models and gains only $+0.5$ of margin for 2.5× the
parameters — cheap, and learned first (§5.5). Factual recall is the opposite: +2.07 → +4.41 and
0.83 → 1.00 accuracy from 160m to 410m, the largest scale effect measured here and consistent with
factual storage being capacity-bound. GPT-2 (124M) outperforms the larger pythia-160m overall, so
parameter count alone does not order these models.

# 6. Analysis and Discussion

## 6.1 What works and what does not

The model is *most* mechanistically legible where it is most reliable. Subject–verb agreement is
at ceiling accuracy in all three models, resolves by mid-stack (§5.2), emerges first and abruptly
during pretraining (§5.5), and depends on a single head whose removal harms nothing else (§5.4) —
four independent measurements converging on one picture. The generative half also works as
intended: a rank-one change to one matrix reliably changes what the model generates without
measurably degrading fluency (Table 6); the model keeps writing coherent English about the edited
subject and simply says something different.

**Factual recall is weak and only weakly localized.** Its top head is half-generic (0.52), its
logit-lens margin resolves late, and pythia-160m ranks the correct capital behind generic
continuations even when its margin is positive. "The model knows the fact" and "the model says the
fact" are different claims; only the first holds at 160M parameters.

**Specificity, not efficacy, is the cost of editing.** Most layers and most penalty settings
achieve efficacy (§5.3, §5.7). Changing one fact and *nothing else* is the hard part: driving
generalization from 0.167 to 0.611 costs all neighbourhood preservation (Table 7). No setting
tested is simultaneously strong, general, and harmless.

**Small models are noisy substrates.** One dead layer (§5.3) moves a 3-target mean by 0.33 — and
several earlier-milestone claims rested on exactly that kind of single-configuration observation.

## 6.2 The methodological lesson

Three of the four novel findings came from one move: taking a result that had been *explained* and
testing the explanation instead. "Causal tracing picks a bad layer, so we need ROME's multi-layer
window" — the window made selection worse (§5.3). "The edit does not generalize because we
simplified ROME's covariance term" — the penalty weight was doing that (§5.7). "GPT-2 resists
editing" — GPT-2 was never edited (§5.7). Each explanation was plausible and cited the right
literature, and each was wrong. What separates them from the claims that survived (L6.H4's
selectivity, the emergence ordering) is that the survivors were measured against a control while
the others were inferred from a single configuration. A hyperparameter tuned on one model is a
*finding about that model*; reporting it as a property of a method is how artifacts become
citations.

## 6.3 Comparison with expectations and prior work

The proposal expected agreement early, factual recall accumulating slowly, and induction appearing
as an abrupt phase change (Olsson et al. [5]). The first two are confirmed (§5.5); the third is
**not** — induction rises steadily here while it is *agreement* that switches on abruptly. The
likely explanation is measurement resolution (11 checkpoints, 16 induction prompts would smooth a
phase change confined to a few thousand steps), but on the evidence collected the abrupt behavior
is agreement's.

Relative to ROME [7], the edit here achieves comparable efficacy on a model roughly 1000× smaller
than GPT-J, with generalization only at the loosened penalty. The identity-covariance
simplification remains untested as an explanation for anything: §5.7 removed the confound credited
to it, so its actual cost is now an open question rather than a settled one.

## 6.4 Limitations

1. **Scale of the evaluation set.** 48 prompts and 3 edit targets against the projected 300–500
   and 20–40. Edit metrics are means over 3 binary outcomes, so single flips move them by 0.33:
   the qualitative claims are robust, the numeric ones are estimates.
2. **Single seed for the edit fits.** The v* optimization is deterministic given a layer, so the
   seed sweep in §5.3 varies tracing noise only. Sampling-based decoding is not evaluated.
3. **Fluency is n-gram entropy**, not perplexity under a reference model, so it detects
   degeneration and repetition but not subtler quality loss.
4. **Floating-point sensitivity.** Log-probabilities differ by ~1e-3 across platforms, enough to
   flip two binary edit scores — again a consequence of 3-target means.
5. **The Pile [11] was not analyzed.** §5.5 answers *when* a behavior appears, not *from what
   text*.

# 7. Conclusions and Future Work

This project set out to open a small generative language model, find the parts responsible for
three behaviors, and use those parts to change what the model writes. All four research questions
were answered; two of the answers are not what was expected.

**Key takeaways.** *(i)* Localization succeeds where behavior is sharp — agreement has a dedicated
head (L6.H4, selectivity 1.00) while factual recall and induction are diffuse, and the largest
ablation effect for every behavior is shared input processing (MLP 0). *(ii)* Editing works, and
its difficulty is specificity, not efficacy: one rank-one update changes what the model generates
at unchanged fluency from most early layers, but no setting produced a strong, paraphrase-general
edit that also left neighbouring facts alone. *(iii)* Behaviors have training histories, and the
last checkpoint is not the best one. *(iv)* Reported failures deserve the scrutiny given to
reported successes — two headline limitations from earlier milestones dissolved under a
hyperparameter sweep, each having had a plausible, well-cited explanation attached.

**What I learned about generative modeling.** A pretrained transformer is genuinely inspectable
with modest tooling and CPU-only compute; observation and causal intervention can disagree, and
the intervention wins; and in a system with this many interacting parts, the gap between "the
result I got" and "the property I claimed" is almost always a control I did not run.

**Future work**, in priority order: (1) **scale the evaluation set** to the proposed 300–500
prompts and 20–40 targets so binary rates become estimates with error bars — highest value, no new
method needed; (2) **make the edit penalty scale-relative**, penalizing
$\lVert\delta\rVert / \lVert v_{\text{orig}} \rVert$, which §5.7 predicts will collapse the three
per-model curves onto one — a directly testable claim; (3) **restore ROME's covariance term** and
measure what the identity simplification actually costs, now that the penalty confound is removed;
(4) **sample checkpoints densely between steps 512 and 8 000**, where all three behaviors move, to
test whether induction's phase change is real and merely under-sampled here; and (5) extend to
**multi-fact editing (MEMIT [8])** to find where specificity degradation becomes prohibitive as
edits accumulate.

# 8. References

[1] A. Vaswani et al., "Attention Is All You Need," *NeurIPS*, 2017.

[2] A. Radford et al., "Language Models are Unsupervised Multitask Learners," OpenAI, 2019.

[3] N. Elhage et al., "A Mathematical Framework for Transformer Circuits," *Transformer Circuits
Thread*, 2021.

[4] nostalgebraist, "Interpreting GPT: the Logit Lens," LessWrong, 2020.

[5] C. Olsson et al., "In-context Learning and Induction Heads," *Transformer Circuits Thread*,
2022.

[6] K. Wang, A. Variengien, A. Conmy, B. Shlegeris, J. Steinhardt, "Interpretability in the Wild:
A Circuit for Indirect Object Identification in GPT-2 Small," *ICLR*, 2023.

[7] K. Meng, D. Bau, A. Andonian, Y. Belinkov, "Locating and Editing Factual Associations in GPT,"
*NeurIPS*, 2022.

[8] K. Meng, A. Sharma, A. Andonian, Y. Belinkov, D. Bau, "Mass-Editing Memory in a Transformer,"
*ICLR*, 2023.

[9] S. Biderman et al., "Pythia: A Suite for Analyzing LLMs Across Training and Scaling,"
*ICML*, 2023.

[10] N. Nanda and J. Bloom, "TransformerLens," GitHub repository, 2022.

[11] L. Gao et al., "The Pile," arXiv:2101.00027, 2020.
