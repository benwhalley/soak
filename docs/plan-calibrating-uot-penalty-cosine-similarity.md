
When you compare two sets of qualitative themes (say, themes from two populations, or from two independent runs of an analysis), you quickly hit an awkward problem: themes are not crisply defined objects. They overlap, one theme can be broader than another, and sometimes a theme in one set simply doesn’t exist in the other. But if you want to talk about “how similar these sets are” you still need some principled way to align them.

Naive set overlap measures like Jaccard don’t really help. Jaccard assumes you have a shared vocabulary of discrete items and you can decide membership cleanly. Theme sets aren’t like that: small wording changes can produce “different” themes that are conceptually the same, and big conceptual differences can hide behind superficially similar labels. Jaccard ends up punishing harmless rephrasings and ignoring near-misses, and it also bakes in the assumption that each theme either matches exactly or not at all.

Hungarian matching (one-to-one assignment) looks more sophisticated but it encodes a similarly rigid idea: each theme must pair with exactly one theme on the other side. That’s often the wrong shape of problem. In real qualitative outputs you frequently get “splits” and “merges”: one broad theme in set A becomes two narrower themes in set B; or two related themes in A get collapsed into one in B. A one-to-one matcher forces an artificial decision: pick one correspondence and discard the rest, or spread the error around arbitrarily. It can look numerically neat while being conceptually misleading.

So people often reach for embeddings and cosine similarity: represent each theme (name + description) as a vector, compute pairwise cosine similarities, and use those numbers to drive matching. This gives you something useful: a *ranking* of which themes are closer. But it doesn’t give you an interpretable *scale*. A cosine similarity of 0.7 might be “basically the same” in one domain and “only loosely related” in another, depending on the embedding model, how you write theme descriptions, and how diverse the corpus is. Even if you transform similarity into something more psychologically plausible (e.g., converting cosine to angular distance, or applying an exponential decay like Shepard’s law), you still haven’t solved the core interpretability issue: the metric has an arbitrary scale. Is 0.3 a small change or a big one? Is 0.7 “similar enough” to treat together? Without calibration, these numbers are great for ordering but weak for decisions. That’s also why most embedding validation work emphasises *rank correlation* with human judgements (do we order pairs correctly?) more than it emphasises absolute similarity calibration (does “0.7” mean the same thing across settings?). Ranking is robust; absolute scale is slippery.

Optimal transport (OT) is attractive here because it matches the *structure* of the theme alignment problem better than one-to-one assignment. Instead of forcing each theme to pick a single partner, OT treats each theme as having some “mass” and allows that mass to be distributed across multiple themes in the other set. That naturally represents splitting and merging: a broad theme can “send” some of its mass to each of several narrower themes, proportional to their semantic closeness. OT also gives you a clean global summary: a single transport cost that reflects how much “work” is needed to turn one theme set into the other.

But vanilla OT has a deal-breaker for qualitative themes: it assumes all mass must be matched. In plain terms, it assumes that everything in set A must correspond to something in set B, and vice versa. That’s not true when a theme is genuinely absent, newly emergent, or just idiosyncratic to one sample. In those cases OT doesn’t say “this theme doesn’t exist over there”; it says “fine, I’ll match it to the least-bad option”. You get forced alignments that look mathematically consistent but conceptually wrong.

Unbalanced optimal transport (UOT) fixes exactly that. It relaxes the “all mass must be matched” constraint and instead allows mass to be created or destroyed at a cost. Now the model can represent “this theme has no counterpart” by letting its mass disappear rather than forcing a bad match. This is much closer to what we actually want: partial correspondence, with explicit permission for non-correspondence.

The catch is that UOT introduces a new knob: the penalty parameter (often written as K, tau, rho, or `reg_m` depending on the formulation/software). This parameter controls how willing the algorithm is to declare something unmatched versus forcing a match. And unless you calibrate it, it’s arbitrary. Set it too high and you’re back to balanced OT (everything gets matched, even nonsense). Set it too low and mass evaporates too readily (you under-match and treat weak-but-real correspondences as absent). Crucially, there is no guarantee that a convenient default value corresponds to what humans would judge as “sufficiently similar to treat together”.

All of this matters because we don’t just want pretty alignments; we want *metrics* that we can use. We’d like to quantify how similar two sets of themes are in order to do things like: compare populations (do these groups talk about their experiences in meaningfully different ways?), assess stability (do repeated LLM runs or different coders produce similar thematic structures?), and detect drift over time. A transport-based distance between theme sets is a good candidate for such a metric -- but only if its behaviour is anchored to a defensible notion of “match”.

So the practical problem becomes: how do we set K?

A simple heuristic is to set K so that themes rarely split into more than, say, three recipients. That matches an intuition about interpretability: if a theme’s mass ends up scattered across ten themes, the mapping is no longer a “split” so much as “noise”. This is a workable engineering constraint, and it may give stable behaviour, but it’s still subjective: why three? why not two or five? And it doesn’t guarantee alignment with human judgements of semantic correspondence.

A more principled route is calibration: choose K so that the model’s match/unmatch behaviour corresponds to typical human judgements about when two pieces of text are “similar enough” to be treated as the same idea.

There are two plausible experiments to achieve this.

Idea 1 (fast, uses existing data): calibrate K using standard semantic similarity datasets. There is a large literature where human annotators rated the semantic similarity of sentence pairs on an ordinal scale (typically 0–5). These datasets (e.g., STS Benchmark and related SemEval STS tasks) were created to evaluate embedding models. The move here is to treat each sentence as analogous to a “theme description” (short text conveying a coherent meaning). We embed each sentence with the same pipeline we use for themes (same model, same preprocessing), compute cosine similarity (or angular distance) for each pair, and learn the mapping between embedding distance and human-rated similarity. That mapping gives us an interpretable anchor: roughly, what cosine/angle values correspond to “borderline similar” according to typical human judgements.

Then we connect that anchor to UOT by interpreting K as a “not worth matching beyond this distance” threshold. In practice: find the embedding distance that corresponds to the mid-point of the human scale (or to whatever region corresponds to ambiguous-but-related pairs), and set K so that, around that distance, UOT is indifferent between matching and leaving unmatched. The result is not a universal truth, but it’s a defensible calibration: K is chosen so that the algorithm’s willingness to match corresponds to a known distribution of human semantic similarity judgements. This approach is attractive because it is quick, reproducible, and can be done without collecting new data. The limitation is obvious: STS datasets are about sentence meaning similarity, not about theme synthesis decisions, and the population of raters (often crowdworkers) may not match your domain. So it gives a good prior and sanity check, but it may not pin down the exact boundary you care about.

Idea 2 (better, slower, task-specific): collect new human judgements on your actual themes, with questions designed to reflect the analytic decision you care about. Instead of asking abstractly “how similar are these?”, you can ask something like: “Would you be comfortable combining these into a single theme when summarising results, without losing something important?” That’s a direct operationalisation of “match”. You can also ask a second question capturing asymmetry: “Does one theme mostly fit inside the other, or do they overlap without containment?” That tells you whether the many-to-many structure is doing meaningful work (splits/merges) rather than just smoothing. This experiment produces exactly what STS cannot: a boundary aligned with your notion of correspondence in thematic analysis. The cost is time: you need to design materials (theme name + description, maybe exemplar quotes), sample pairs carefully (not random pairs -- you need many borderline pairs), and recruit appropriate raters (ideally people with enough context to judge the themes).

Right now, if you only have time for one thing, idea 1 is the pragmatic choice. It won’t fully solve “what counts as a match for theme synthesis”, but it can do something valuable immediately: it can turn cosine/angle numbers from “mysterious scale” into “numbers with a known relationship to typical human semantic similarity ratings”, and it can give you a principled starting point for K rather than a vibes-based one. If later you have time for idea 2, you can treat the STS-calibrated K as a prior and then refine it with task-specific judgements.

If you want, next I can (without going full methods) suggest which existing similarity datasets are most structurally similar to “theme name + description”, and what the calibration output should look like (e.g., a distance band corresponding to “borderline similar” plus an uncertainty range, rather than a single magic K).










https://arxiv.org/pdf/1708.00055



MedSTS
https://arxiv.org/pdf/1808.09397
Examples:
The two sentences are not equivalent, but share some details.
S1àDiscussed risks, goals, alternatives, advance directives, and the necessity of other members of the healthcare team participating in the procedure with (patient) (legal representative and others present during the discussion).
S2 à We discussed the low likelihood that a blood transfusion would be required during the postoperative period and the necessity of other members of the surgical team participating in the procedure.