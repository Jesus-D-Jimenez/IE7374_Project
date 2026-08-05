# Presentation script — Group 25, "Opening the Black Box"

Read-aloud narration for the 20-slide deck. **1,427 spoken words**, and the per-slide timestamps
below are computed from the actual word counts, not estimated — re-run
`python docs/report/time_script.py --write` after any edit and they update themselves.

| Pace | Full script | Dropping the two *[optional cut]* paragraphs |
|---|---|---|
| 140 wpm (slow, deliberate) | 10:11 — **over the cap** | 9:43 |
| 150 wpm (normal reading) | 9:30 | 9:04 |
| 160 wpm (brisk) | 8:55 | 8:30 |

**Delivery notes**

* Aim for 150 wpm or a touch faster. If you read deliberately, drop the two paragraphs marked
  *[optional cut]* (slides 7 and 13) and you stay under 10 minutes at any pace.
* Say the **bolded numbers** exactly — they are the evidence the "clear explanations" band wants.
* On every figure slide, name the axes before the finding: "what you're looking at is X against
  Y," then "and the thing to notice is…". That habit is most of the difference between *showing*
  a visualization and *communicating* one.
* Slides 15 and 17 carry the strongest result — slow down there. Speed up on 3, 4 and 9.
* Don't read the bullets. The bullets are for the viewer; this is for you.
* Record with PowerPoint's *Slide Show → Record*; it exports video directly. Then
  `python scripts/package_submission.py --video <file>` folds it into the submission ZIP.

---

## Slide 1 — Title *(0:00–0:19)*

I'm Jesus Jimenez, and this is my final project: "Opening the Black Box."

The one-sentence version: I take a small language model apart, find the pieces responsible for
specific behaviors, and then use those pieces to change what the model writes — by editing exactly
one weight matrix.

## Slide 2 — The problem *(0:19–0:57)*

We normally treat a language model as a black box. Text in, text out, and the only handle is the
prompt.

That's expensive. When a model states a fact that's wrong, the options are retraining or
fine-tuning — both change the entire network to correct one thing.

So I asked the opposite question. Taking **Pythia-160m**, a **162-million** parameter open model:
can I find *where* a behavior lives and change it surgically? Nothing is trained or fine-tuned —
the only weight change anywhere is one rank-one update, and it runs on a laptop CPU in a minute.

## Slide 3 — Four research questions *(0:57–1:27)*

**One:** which attention heads and MLPs causally drive subject-verb agreement, factual recall,
and induction. **Two:** when I switch those off, does performance drop *selectively* — selectivity
being what separates a real mechanism from a coincidence.

**Three**, the generative core: can that localization edit the model so it *generates* text
consistent with a new fact, and does that text survive the standard evaluation?

**Four:** Pythia publishes **154 checkpoints** from training — when does each behavior appear?

## Slide 4 — Data *(1:27–1:58)*

The data is hand-built, because the measurement needs matched pairs public benchmarks don't give
you. **48 prompts**, each with a correct continuation and a matched incorrect one differing in
exactly the property being tested — "the keys on the table" is followed by "are," not "is."

Plus **three edit targets**, each with two paraphrases and two neighbouring facts that must *not*
change. Up front: 48 prompts is small, and it's the main limitation of this work.

## Slide 5 — Method *(1:58–2:40)*

Three steps.

**Measure.** Every claim reduces to one number — the logit difference, the model's score for the
correct answer minus its score for the wrong one.

**Localize**, with three tools of increasing strength. The logit lens reads each layer.
Zero-ablation switches a component off and measures the damage. Causal tracing corrupts the
subject, then restores clean activations layer by layer to find where a fact is stored.

**Edit and generate.** I optimize a vector at the subject's last token, then apply a rank-one
update to that MLP's output matrix — this is ROME — and score the *generated text* on efficacy,
generalization, specificity and fluency.

## Slide 6 — Ablation heatmap *(2:40–3:05)*

First result. Each panel is one behavior. Rows are layers, columns are attention heads, and the
colour is how much the logit difference drops when I switch that head off — darker red means the
model needed it more.

Look at the agreement panel on the left. One cell dominates: **layer 6, head 4**. Nothing like it
in the other two panels.

## Slide 7 — Selectivity *(3:05–3:43)*

That's the number that matters. Ablating **L6.H4** costs **2.29** logits of agreement margin, and
on the other two behaviors it costs **negative 0.02** — nothing measurable. Selectivity **1.00**.
A dedicated component, not a shared pathway that happens to matter.

Compare factual recall at **0.52** — that same head is also worth nearly a full logit to
agreement. That's a diffuse mechanism.

*[optional cut]* One honest caveat: the biggest component for all three behaviors isn't a head at
all. It's **MLP 0**, and ablating it hurts everything — which is why "largest drop" and "the
mechanism" are different claims.

## Slide 8 — The edit works *(3:43–4:14)*

Now the generative half. Same model, same prompt, same greedy decoding — the only difference
between these two generations is one weight matrix.

Before: "Mount Everest is located in the country of **Nepal**, the mountain is located in the
Himalayas…" After: "…located in the country of **Canada**. The Mount Everest National Park is
located in the country of Canada…"

It doesn't just swap a word — it keeps writing coherent text *around* the new fact, at unchanged
fluency.

## Slide 9 — Three models *(4:14–4:39)*

That same evaluation across three models gives a tidy story. GPT-2: efficacy **zero**, the edit
never takes. Pythia-160m: **0.67**. Pythia-410m: **1.00**, and the only one where the edit
survives a paraphrase, at **0.50**.

The obvious conclusion is that editing works better at scale and not at all on GPT-2. Hold that
thought — in three minutes I'll show you it's wrong.

## Slide 10 — Emergence figure *(4:39–4:58)*

First, question four — the one thing you can only do with Pythia, because it publishes
checkpoints from during training.

Training step on the x-axis, log scale, from random initialization out to the final step
**143,000**. Strength of each behavior on the left, accuracy on the right.

## Slide 11 — Three shapes *(4:58–5:38)*

Three behaviors, three completely different shapes.

**Agreement** switches on abruptly — a **13-fold** jump between steps 512 and 1,000 — and is
finished by step 4,000. **Induction** climbs steadily across an order of magnitude of training.

**Factual recall** does something I didn't expect: it's *worse than chance* around steps 128 to
512. The model has learned that "the capital of X is" is followed by a city, and confidently picks
the wrong one, before it learns which.

And at the right-hand end, for two of three behaviors, **step 64,000 beats the final step
143,000**. The finished model is not the best model.

## Slide 12 — Layer sweep figure *(5:38–6:03)*

Now the part I'm most pleased with, and it starts with me doubting my own earlier write-up.

An earlier milestone of mine said causal tracing sometimes picks a bad layer, and the fix is
ROME's multi-layer window. Rather than assume that, I tested it — every target at *every* layer,
**36** separate fits. The shaded bands are the layers tracing actually picks.

## Slide 13 — The fix made it worse *(6:03–6:26)*

The result: the raw rule I already had scores **0.933**. The windowed "fix" scores **0.733**. The
proposed improvement selects *worse* layers than the thing it was meant to replace.

*[optional cut]* So it's implemented, but it isn't the default — and this measurement is why.
That's a decision I can defend with a number instead of a citation.

## Slide 14 — Why the layer mattered less *(6:26–6:49)*

The deeper finding: layer choice barely matters. Most early layers hold the edit — for Mount
Everest, **9 of 12**.

But the last three never work, which lines up exactly with the logit lens: by layer 9 the answer
is already decided, so an edit arrives too late. Two different methods agreeing on where the
decision gets made.

## Slide 15 — The GPT-2 mystery *(6:49–7:30)*

Back to GPT-2, which supposedly couldn't be edited.

When I watched the optimization, the loss was *going up*. The injected vector stalled at magnitude
**2.76**, where Pythia reaches 6 or 7.

Here's why. The objective penalizes that vector's size in *absolute* terms — but how big it needs
to be depends on the residual stream you add it to, and those aren't comparable across models. At
the edited layer GPT-2's residual norm is **75**; Pythia's is **13**. Six times, and squared.

GPT-2's edit was never attempted — it was regularized out of existence. Loosen that one number and
GPT-2 reaches efficacy **1.00**.

## Slide 16 — Penalty figure *(7:30–7:47)*

Once I swept that hyperparameter, a second thing fell out. Penalty weight on the x-axis, the three
quality scores on the y-axis, pooled across all three models. On the right, how large an edit each
setting permits.

Watch the red and green lines cross.

## Slide 17 — A trade-off, not a property *(7:47–8:19)*

My earlier write-up reported generalization of **zero** — edits never surviving a paraphrase — and
blamed a simplification in the method.

Also wrong. As the penalty falls, generalization climbs from **0.167** to **0.611** while
preservation of neighbouring facts collapses from **0.944** to **zero**. Not a property of the
method — one end of a trade-off a single configuration couldn't reveal.

I left the committed default alone, so the trade-off is documented rather than quietly re-tuned
into a better-looking number.

## Slide 18 — What I learned *(8:19–8:47)*

Localization succeeds where behavior is sharp. Editing's hard part is specificity, not efficacy —
anyone can change a fact; changing *only* that fact is the problem. Behaviors have training
histories, and the last checkpoint isn't the best one.

And the one I'd take forward: reported failures deserve the scrutiny we give reported successes.
Two of my own documented limitations dissolved under one sweep, and both had a plausible,
well-cited explanation attached.

## Slide 19 — Limits and next *(8:47–9:14)*

Honestly: 48 prompts and 3 edit targets, so every edit metric is a mean over three binary
outcomes. The qualitative claims are solid; the decimals are estimates.

Next: scale the evaluation set so those become error bars; make the penalty *scale-relative*,
which predicts the three per-model curves collapse onto one — a testable claim, not a wish; then
restore the full covariance term now the confound is gone.

## Slide 20 — Close *(9:14–9:29)*

All of it reproduces from one command in about a minute on CPU, with 38 tests, a container, and
every figure in this talk regenerating from the repository.

Thank you — happy to take questions.

---

# Q&A preparation

The rubric's top band asks for questions answered *thoughtfully, demonstrating deep understanding*.
These are the questions this work invites. Lead with a number, then the reasoning.

**"Isn't 48 prompts and 3 edit targets too small to conclude anything?"**
For the decimals, yes — one edit flipping moves a 3-target mean by 0.33, and the report says so.
What sample size can't explain away are the qualitative results: L6.H4's selectivity is a
2.29-versus-negative-0.02 gap, and the emergence ordering holds across 48 prompts at eleven
checkpoints. Scaling the prompt set is my first next step precisely because it turns the weak
claims into measurable ones without needing a new method.

**"Why not just fine-tune to fix the fact?"**
Fine-tuning updates every weight for one correction, needs data, and gives no guarantee about what
else moved. The rank-one edit touches one matrix, takes 25 steps, and comes with a specificity
metric that tells you what it broke. Though my own results show that specificity guarantee is the
weak part — which is the honest counterargument to my approach.

**"How do you know L6.H4 is real and not an artifact of your prompts?"**
Two things: it's an *intervention*, not a correlation — I switch the head off and the behavior
degrades — and it's measured against a control, the same ablation on the other two behaviors
costing nothing. That said, it's four prompts per behavior, so I'd want the larger suite before
calling it "the agreement head" in print.

**"n-gram entropy is a weak fluency metric."**
It is. It catches degeneration and repetition, which is what a bad edit usually produces, and it's
model-free so the edited model can't game it. But it won't catch subtle quality loss — perplexity
under an independent reference model would be stronger.

**"If a smaller penalty gives better edits, why not change the default?"**
Because "better" depends which column you read. Dropping the penalty raises generalization to
0.611 and takes neighbourhood preservation to zero — the edit starts bleeding into unrelated
facts. No setting in my sweep is simultaneously strong, general and harmless, so I left the
default at the conservative end and documented the trade-off instead of picking the number that
flatters the report.

**"What would you do differently?"**
Sweep the hyperparameters *before* writing any explanation for a failure. Both limitations I had
to retract were explanations I reached for instead of running a control that cost twenty minutes
of compute.

**"Does this scale to a real model?"**
The method does — ROME was demonstrated on GPT-J, about a thousand times larger. What doesn't
transfer is the configuration, which is exactly what the GPT-2 result is about: the same absolute
penalty means different things in models with different residual-stream norms. Scale-relative
regularization is the first thing I'd test.
