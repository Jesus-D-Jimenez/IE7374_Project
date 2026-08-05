# Presentation script — Group 25, "Opening the Black Box"

Read-aloud narration for the 20-slide deck, written to sound like me talking, not like a paper
being read out loud. **828 words — 5:31 at a normal reading pace, 5:10 if you're brisk.**
Timestamps come from the actual word counts, so re-run
`python docs/report/time_script.py --write` after any edit and they update themselves.

**Delivery notes**

* About 16 seconds a slide. Keep moving.
* Say the **bolded numbers** exactly. Everything else you can put in your own words; this is a
  script to know, not to recite.
* On the figure slides (6, 10, 12, 16) say what the axes are *before* you say what the result is.
  That habit is most of the difference between showing a chart and explaining one.
* Slides 15 and 17 are the best part of the project. Slow down there.
* Don't read the bullets out loud — the viewer can already see them.
* Record with PowerPoint's *Slide Show → Record*, then
  `python scripts/package_submission.py --video <file>` puts it in the submission ZIP.

---

## Slide 1 — Title *(0:00–0:14)*

Hey, I'm Jesus. My project is Opening the Black Box.

I took a small language model apart, found the parts that do specific things, and edited one of
them to change what the model writes.

## Slide 2 — The problem *(0:14–0:36)*

A language model is basically a black box — text in, text out, and all you control is the prompt.

So when it says something wrong, your only options are retraining or fine-tuning, and both change
the whole network to fix one fact. I wanted to find where one fact lives and change just that.

## Slide 3 — Four research questions *(0:36–0:54)*

Four questions. Which heads and MLPs cause each behavior. Whether turning them off breaks *only*
that behavior. Whether I can use that to edit a fact so the model generates it. And since Pythia
saves **154 checkpoints** while training — when each behavior shows up.

## Slide 4 — Data *(0:54–1:13)*

I wrote the data myself. **48 prompts**, each with a right answer and a wrong answer that differ
in just the thing I'm testing — "the keys on the table" should be "are," not "is." Plus three
facts to edit.

48 is small, and that's my biggest limitation.

## Slide 5 — Method *(1:13–1:31)*

Three steps. **Measure**: how much the model prefers the right answer. **Localize**: turn off one
head at a time and see what breaks, and use causal tracing to find where a fact is stored.
**Edit**: change one weight matrix, then score what the model actually writes.

## Slide 6 — Ablation heatmap *(1:31–1:48)*

This is every attention head. Rows are layers, columns are heads, darker red means the model
needed that head more.

Look at the agreement panel on the left — one square is way darker than the rest. **Layer 6,
head 4**.

## Slide 7 — Selectivity *(1:48–2:05)*

Turning off **L6.H4** costs **2.29** on agreement and **negative 0.02** on the other two
behaviors — basically nothing. It's not a general-purpose head, it's doing agreement
specifically.

Factual recall has nothing like that. Its best head scores **0.52**, because it helps with
everything.

## Slide 8 — The edit works *(2:05–2:27)*

Now the editing. Same model, same prompt, and the only difference is one weight matrix.

Before: "Mount Everest is located in the country of **Nepal**." After: "…the country of
**Canada**," and then it keeps writing about the park being in Canada.

So it's not just swapping a word — it writes around the new fact.

## Slide 9 — Three models *(2:27–2:37)*

Same edit on three models. GPT-2: **zero**, never works. Pythia-160m: **0.67**. Pythia-410m:
**1.00**.

So editing looks like it just works better on bigger models. Remember that.

## Slide 10 — Emergence figure *(2:37–2:46)*

Training step across the bottom, random weights out to the end of training, and how strong each
behavior is going up.

## Slide 11 — Three shapes *(2:46–3:07)*

All three show up differently. Agreement jumps **13 times** between step 512 and 1,000, then it's
done. Induction comes in slowly.

Factual recall is *worse than random* early on — the model learns a city comes next before it
learns which city. And **step 64,000 beats the final step** on two of three behaviors.

## Slide 12 — Layer sweep figure *(3:07–3:23)*

This is the part I like. In an earlier milestone I wrote that tracing sometimes picks a bad
layer, and the fix is a window over several layers.

So I tested it — every fact at *every* layer, **36** runs.

## Slide 13 — The fix made it worse *(3:23–3:36)*

The fix was worse. My original rule scores **0.933**; the "better" one scores **0.733**.

So I kept the original, and now I can explain that with a number instead of a citation.

## Slide 14 — Why the layer mattered less *(3:36–3:49)*

The layer barely matters either — most early layers work. But the last three never do, because by
layer 9 the model already picked its answer. Editing after that is too late.

## Slide 15 — The GPT-2 mystery *(3:49–4:20)*

Back to GPT-2. When I watched the optimization, the loss was going *up* and the vector it added
stayed tiny — **2.76**, where Pythia gets to 6 or 7.

The penalty on that vector is a fixed number, but how big it needs to be depends on the model.
GPT-2's activations are around **75**; Pythia's are **13**. Same penalty, way harsher.

I never actually edited GPT-2 — I squashed it. Change that setting and it gets **1.00**.

## Slide 16 — Penalty figure *(4:20–4:28)*

So I swept that setting. Penalty across the bottom, the three scores going up. Watch the red and
green lines cross.

## Slide 17 — A trade-off, not a property *(4:28–4:48)*

I'd also written that edits never survive rewording, and blamed the method. Also wrong.

Lower the penalty and generalization goes from **0.167** to **0.611**, but protecting other facts
drops from **0.944** to **zero**. It's a trade-off, not a broken method — so I left the default
alone and documented it.

## Slide 18 — What I learned *(4:48–5:06)*

You can find a behavior when it's sharp, like agreement. Editing a fact is easy; editing *only*
that fact is hard. Models get better and worse at things while training.

And the big one: two of my own limitations turned out to be my own settings.

## Slide 19 — Limits and next *(5:06–5:20)*

Limits: 48 prompts and 3 facts, so my decimals are estimates. Next I'd scale the data up, make the
penalty scale with the model, and add back the part of the method I simplified.

## Slide 20 — Close *(5:20–5:31)*

All of it reruns with one command in about a minute, there are 38 tests, and it's all on GitHub.
Thanks — happy to take questions.

---

# Q&A preparation

Not part of the five minutes. Lead with a number, then the reason, and don't get defensive about
the small dataset — owning it reads better than dodging it.

**"Isn't 48 prompts and 3 edit targets too small?"**
For the decimals, yeah. One edit flipping moves a 3-target average by 0.33, and I say that in the
report. What the size *doesn't* explain away is the big gaps — L6.H4 is 2.29 versus negative 0.02,
and the training-order result holds across all 48 prompts at eleven checkpoints. Scaling the data
is my first next step for exactly that reason.

**"Why not just fine-tune the model to fix the fact?"**
Fine-tuning changes every weight for one correction, needs training data, and doesn't tell you
what else moved. The edit I use touches one matrix, takes 25 steps, and comes with a metric for
what it broke. Although honestly, my own results show that metric is the weak part.

**"How do you know L6.H4 is real and not just your prompts?"**
Two reasons. It's an intervention, not a correlation — I turn the head off and the behavior gets
worse. And I checked it against a control: the same head costs nothing on the other two behaviors.
But it's four prompts per behavior, so I wouldn't call it "the agreement head" in a paper yet.

**"Your fluency metric is just n-gram entropy."**
Right, and it's the weakest metric I have. It catches repetition and gibberish, which is what
broken edits usually produce, and the model can't game it. But it won't catch subtle quality loss.
Perplexity under a different model would be better.

**"If a lower penalty gives better edits, why not change the default?"**
Because "better" depends which column you look at. Lower penalty gets generalization up to 0.611
but takes protection of other facts to zero — the edit starts leaking into unrelated facts.
Nothing in my sweep was strong, general, *and* safe at the same time, so I left the default at the
safe end and wrote up the trade-off instead of picking the number that looks best.

**"What would you do differently?"**
Test the hyperparameters before writing an explanation for a failure. Both things I had to take
back were explanations I came up with instead of running a check that took twenty minutes.

**"Does this work on a real, large model?"**
The method does — ROME was done on GPT-J, about a thousand times bigger. What doesn't carry over
is the settings, which is the whole point of my GPT-2 result: the same fixed penalty means
different things in models with different activation sizes. Making it scale-relative is the first
thing I'd try.
