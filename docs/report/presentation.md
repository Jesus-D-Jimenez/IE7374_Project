% Opening the Black Box
% Jesus D. Jimenez Ballestas · Group 25 · IE7374 Generative AI
% Locating, ablating, and editing the mechanisms of a small language model

# The problem

- A language model is usually a black box: prompt in, text out
- When it states a **wrong fact**, the conventional fixes are retraining or fine-tuning — both
  change the entire network to correct one thing
- **This project:** treat EleutherAI's **Pythia-160m** as a system with parts
    - Find the parts responsible for three specific behaviors
    - Use that to **change what the model writes** by editing *one weight matrix*
    - Nothing trained, nothing fine-tuned — one deliberate rank-one update
    - Runs on a laptop CPU in about a minute

::: notes
Interpretability is not the goal here — control is. If you can find where a fact lives, you can
fix it surgically instead of retraining.
:::

# Four research questions

- **RQ1** — which attention heads and MLPs *causally* drive subject–verb agreement, factual
  recall, and induction?
- **RQ2** — when we ablate them, does performance drop **selectively**?
- **RQ3 (generative core)** — can that localization be used to edit the model so it **generates**
  the new fact, with efficacy, generalization, specificity and fluency?
- **RQ4** — using Pythia's 154 training checkpoints, **when** does each behavior appear?

::: notes
RQ3 is what makes this a generative project rather than only an analysis project: the model
produces output under the edit, and that output is what gets scored.
:::

# Data: minimal pairs, hand-built

| Suite | Prompts | Example | Correct | Incorrect |
|---|---:|---|---|---|
| Agreement | 20 | `The keys on the table` | ` are` | ` is` |
| Factual recall | 12 | `The capital of France is` | ` Paris` | ` London` |
| Induction | 16 | `AliceBob Alice` | `Bob` | `Carol` |

::: notes
48 prompts, plus 3 edit targets each with two paraphrases and two neighbourhood facts — Eiffel
Tower, Paris to Rome. Every answer is verified single-token so the probability comparison is
clean. 48 is small, and that is the project's main limitation.
:::

# Method: measure → localize → edit

- **Measure** — logit difference: logit(correct) − logit(incorrect), one forward pass
- **Localize**
    - logit lens — where the answer forms (observational)
    - zero-ablation of every head and MLP — causal contribution
    - causal tracing — corrupt the subject, restore layer by layer
- **Edit (ROME-style)** — optimize a delta at the subject's last token, then a closed-form
  rank-one update to that MLP's output matrix
- **Score the generated text** — efficacy · generalization · specificity · fluency

::: notes
Two metric families matching the two halves of the project. One command runs all of it:
python src/model_runner.py, about 60 seconds on CPU.
:::

# Result 1 — ablating every head, per behavior

![](outputs/study/ablation_heatmaps.png){width=9in}

::: notes
Each cell is the drop in logit difference when that head is zeroed, averaged over four prompts.
One dark cell dominates the agreement panel: layer 6, head 4.
:::

# One head owns subject–verb agreement

| Behavior | Top head | Own behavior | Other behaviors | **Selectivity** |
|---|---|---:|---:|---:|
| Agreement | **L6.H4** | +2.29 | −0.02 | **1.00** |
| Induction | L0.H0 | +1.19 | +0.17 | 0.75 |
| Factual recall | L0.H7 | +1.89 | +0.60 | 0.52 |

::: notes
Selectivity 1.00 means ablating L6.H4 costs 2.29 logits of agreement margin and nothing measurable
anywhere else — a dedicated component. Factual recall is diffuse by comparison. Worth noting: the
largest single component for all three behaviors is MLP 0, which is shared input processing, not a
mechanism. Largest drop and "the mechanism" are different claims.
:::

# Result 2 — the edit changes what it writes

*Pythia-160m · Mount Everest: Nepal → Canada · one weight matrix differs*

**before** — *Mount Everest is located in the country of Nepal. The mountain is located in the
Himalayas, and is the highest mountain in the world…*

**after** — *Mount Everest is located in the country of Canada. The Mount Everest National Park is
located in the country of Canada…*

::: notes
Fluency is unchanged — the model keeps writing coherent English about the edited subject, it just
says something different. This is RQ3 answered in the affirmative.
:::

# The same edit in three models

| Model | Efficacy | Generalization | Specificity | Top-1 preserved |
|---|---:|---:|---:|---:|
| gpt2 | 0.00 | 0.00 | 0.33 | 1.00 |
| pythia-160m | 0.67 | 0.00 | 0.33 | 0.83 |
| pythia-410m | **1.00** | **0.50** | 0.50 | 1.00 |

::: notes
The tidy story: editing works better at scale and not at all on GPT-2. Two slides from now I show
that story is wrong — and why testing it was the most valuable thing I did.
:::

# Result 3 — when do behaviors appear?

![](outputs/study/checkpoint_emergence.png){width=9.5in}

::: notes
Eleven Pythia checkpoints from random init to fully trained, all 48 prompts at each.
:::

# Three behaviors, three shapes

- **Agreement** switches on abruptly — 13× jump between steps 512 and 1 000, done by step 4 000
- **Induction** climbs steadily over an order of magnitude of training
- **Factual recall** is *worse than chance* at steps 128–512: right answer *type*, wrong answer —
  it learns "a city follows this prompt" before it learns which city
- **Step 64 000 beats the final step 143 000** on two of three behaviors

::: notes
A behavior can get worse on its way to getting better — something a single end-of-training
measurement cannot see. This is only possible because Pythia publishes checkpoints.
:::

# Result 4 — testing our own explanation

![](outputs/study/layer_sweep.png){width=9.5in}

::: notes
Earlier milestone claimed: causal tracing picks a bad layer, and the fix is ROME's multi-layer
window. So I edited every target at every layer — 36 rank-one fits — to find out what the rule
should have picked.
:::

# The proposed fix made it worse

| Layer-selection rule | Mean efficacy | Mean top-1 preserved |
|---|---:|---:|
| raw argmax (current default) | **0.933** | 0.800 |
| windowed argmax (w=3) | 0.733 | 0.500 |
| windowed argmax (w=5) | 0.800 | 0.667 |

::: notes
The raw rule lands on a working layer in 14 of 15 seed-and-target combinations. The windowed rule
is implemented but is not the default, and this measurement is why.
:::

# Why the layer mattered less than we thought

- Most **early** layers hold the edit — Everest works from 9 of 12 layers
- The last three layers **never** work: by layer 9 the answer is already decided, so the edit
  arrives too late — exactly what the logit lens showed independently
- The original failure was **one isolated dead layer**, not a depth effect

::: notes
Two independent methods — logit lens and the layer sweep — agreeing on where the decision is made
is the strongest evidence in the project after the L6.H4 result.
:::

# Result 5 — the GPT-2 mystery

- During the fit on GPT-2 the loss *increased*; ‖δ‖ stalled at **2.76**, where Pythia reaches 6–7
- The objective penalizes ‖δ‖ in **absolute** terms — but the δ a model needs scales with its
  residual-stream norm
- Residual norm at the edited layer: **gpt2 75.0** · pythia-160m 13.4 · pythia-410m 18.9
- GPT-2's edit was never attempted — it was **regularized out of existence**
- At `edit_kl_weight = 0.001`, GPT-2 reaches efficacy **1.00**

::: notes
This is the finding I am most pleased with, and it came from refusing to accept a plausible
explanation. Nothing about GPT-2 resists editing; the hyperparameter was tuned on Pythia.
:::

# The trade-off that was hidden

![](outputs/study/edit_penalty.png){width=9.5in}

::: notes
Left: pooled over three models and three targets, as the penalty falls, efficacy and
generalization rise while neighbourhood preservation collapses. Right: the delta size each
penalty permits, per model.
:::

# "Generalization = 0.00" was a setting, not a property

| `edit_kl_weight` | Mean ‖δ‖ | Efficacy | Generalization | Top-1 preserved |
|---|---:|---:|---:|---:|
| 0.0625 (default) | 5.1 | 0.556 | 0.167 | **0.944** |
| 0.01 | 12.4 | **0.889** | 0.444 | 0.722 |
| 0.001 | 29.6 | **0.889** | 0.500 | 0.389 |
| 0 | 91.3 | 0.778 | **0.611** | 0.000 |

::: notes
The committed default was deliberately not re-tuned to the better-looking setting, so the pipeline
still reports the conservative configuration and the trade-off is documented rather than hidden.
At 0.01, pythia-410m gets efficacy 1.00, generalization 0.83 and top-1 preserved 0.83 at once.
:::

# What I learned

- **Localization succeeds where behavior is sharp** — one head for agreement, diffuse machinery
  for facts
- **Editing's hard part is specificity, not efficacy** — most layers and most settings can change
  a fact; none changed *only* that fact
- **Behaviors have training histories** — and the last checkpoint is not the best one
- **Reported failures deserve the scrutiny given to successes** — two headline limitations
  dissolved under one hyperparameter sweep, each with a plausible, well-cited explanation attached

::: notes
The methodological point is the one I would want an audience to leave with: a hyperparameter tuned
on one model is a finding about that model, not a property of a method.
:::

# Limits and what's next

- **Limits** — 48 prompts and 3 edit targets, so every edit metric is a mean over 3 binary
  outcomes; n-gram-entropy fluency; single seed per fit; the Pile itself was never analyzed
- **Next**
    1. Scale to 300–500 prompts / 20–40 targets — error bars, not point estimates
    2. Make the penalty **scale-relative** — predicted to collapse the three per-model curves
       onto one, a falsifiable claim
    3. Restore ROME's covariance term, now that the confound is removed
    4. Denser checkpoints between steps 512 and 8 000
    5. Multi-fact editing (MEMIT)

::: notes
Point 2 comes directly out of the penalty experiment and is testable in an afternoon.
:::

# Reproduce everything

```bash
git clone https://github.com/Jesus-D-Jimenez/IE7374_Project
pip install -r requirements.txt
python src/model_runner.py      # 10 samples + 3 edits, ~60 s on CPU
pytest -q                       # 38 tests
```

- Every figure in this talk regenerates from `scripts/`
- Committed outputs, run metadata with package versions, Dockerfile, CI
- Full tables and numbers: `outputs/study/README.md`

**Questions?**

::: notes
Same seed and machine reproduces the committed numbers exactly; the container reproduces every
generation across platforms.
:::
