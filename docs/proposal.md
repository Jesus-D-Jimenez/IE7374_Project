# Project Proposal (summary)

The full proposal is [`Group25_Final_Proposal.docx`](Group25_Final_Proposal.docx). Short version:

**Title.** Opening the Black Box: How and Why a Generative Language Model Makes Its Predictions.

**Idea.** Take a small open generative model (EleutherAI **Pythia** 160M/410M, with **GPT-2 small**
as a control), locate the internal components that drive specific behaviors, then **use those
components to edit the model and generate text under the edit** — turning a purely analytical study
into one that produces and evaluates model outputs.

**Research questions.**
1. Which attention heads / MLP layers causally drive subject–verb agreement, factual recall, and
   induction?
2. Does ablating them hurt the targeted behavior *selectively*?
3. *(generative core)* Using the localized factual-recall components, can a rank-one edit make the
   model **generate** text consistent with a changed fact — with good efficacy, generalization,
   specificity, and fluency?
4. *(stretch)* Using Pythia's 154 training checkpoints, when do these components emerge?

**Experiments.** (1) baseline logit differences → (2) attention + logit lens → (3) ablation +
activation patching + causal tracing → (4) ROME-style edit → generate → evaluate → (5) GPT-2
replication.

See [`methods_and_literature_review.md`](methods_and_literature_review.md),
[`benchmarking.md`](benchmarking.md), and [`model_documentation.md`](model_documentation.md) for
detail.
