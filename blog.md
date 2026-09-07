# The Interference Budget

### How a memory that never grows still forgets — and why that is the whole story of linear attention and BDH

*DataForge 2026 · Pathway Track · companion blog to the interactive explainer*

---

## One sentence, and a dare

Here is the sentence the explainer teaches:

> A fixed-size fast-weight memory `S = Σ kᵢvᵢᵀ` stores a stream of key→value
> bindings without ever growing, but because every binding is added into the same
> matrix, reading back an earlier value returns the true value plus a cross-talk
> term `Σⱼ≠ᵢ (kᵢ·kⱼ)vⱼ` whose size is set by how much the keys overlap — so recall
> error grows like `√(N/d)` and is near-zero only when the keys are
> near-orthogonal.

It is falsifiable. If you can keep recall sharp while packing many *correlated*
keys into the memory, the sentence is wrong. The interactive artifact gives you
every knob you need to try. This post explains where the sentence comes from, why
each clause is true, and why it is the right lens for understanding Pathway's
Dragon Hatchling (BDH).

---

## 1. Two ways to remember the past

Softmax attention remembers by *keeping*. Every token leaves a `(key, value)` pair
in a cache; answering a query means comparing the query to every stored key. The
cache grows with sequence length, and so do the memory footprint and the
per-token compute. This is the well-known cost of long context, and the reason a
small industry exists around KV-cache eviction, compression and retrieval.

A **fast-weight** memory remembers by *folding*. It never stores individual pairs.
It keeps one matrix `S` and, for each binding `(kᵢ, vᵢ)`, adds the rank-1 outer
product into it:

```
Sᵢ = Sᵢ₋₁ + kᵢ vᵢᵀ           (S is d×m, fixed size, independent of i)
```

This is a Hebbian rule: patterns that are active together get their connection
strengthened. Schmidhuber proposed exactly this "fast weight" scheme in 1992; Ba,
Hinton and colleagues revived it in 2016. In the explainer's §1 you can step
through twenty writes and watch `S` fill in — after twenty it looks like noise,
but every binding is still in there, folded rather than filed.

To read the value for a key `q`:

```
out = Sᵀ q = Σⱼ (kⱼ · q) vⱼ
```

Constant memory, linear-time inference. The catch is in the next section.

---

## 2. What you get back is signal plus cross-talk

Set `q = kᵢ`, one of the keys you stored, and split the sum. With unit-norm keys
`kᵢ·kᵢ = 1`:

```
out = vᵢ  +  Σⱼ≠ᵢ (kᵢ · kⱼ) vⱼ
      ────     ──────────────────
      signal   cross-talk
```

The cross-talk is not noise the system injected. It is *the other stored values,
leaking in through the dot products between their keys and yours*. If every other
key is exactly perpendicular to `kᵢ`, every `kᵢ·kⱼ = 0` and recall is exact. As
keys tilt toward each other, their values bleed into your answer.

The explainer's §2 draws the true `vᵢ` and the returned `Sᵀkᵢ` as side-by-side
bars, plus the analytic cross-talk vector. The gap between the bars *is* the
cross-talk vector — we are not estimating the error, we are decomposing it.

**A common misconception, checked.** "A bigger state means a longer memory." A
`d×m` state has only `min(d, m)` independent directions. Beyond that, new writes
must reuse directions already spoken for, and interference is forced by linear
algebra, not by a design choice you can tune away.

---

## 3. The law: `√(N/d)`

The cross-talk term is a sum of `N−1` roughly independent random vectors, each
scaled by a dot product `kᵢ·kⱼ`. For random unit keys in `d` dimensions,
`E[(kᵢ·kⱼ)²] ≈ 1/d`. Values have unit norm. So the expected squared error is
about `(N−1)/d`, and:

```
RMS recall error  ≈  √( (N−1) · ( ρ² + (1−ρ²)/d ) )   →   ≈ √(N/d)  when ρ = 0
```

where `ρ` is the average pairwise key correlation. The explainer's §3 sweeps `N`
or `d`, runs dozens of real trials per point, and plots the measured RMS error
against this predicted curve. At `ρ = 0` the dots sit on the line (mean absolute
deviation ≈ 0.03). Raise `ρ` and the measured curve lifts off: the `ρ²` term
inside the square root becomes a floor that no amount of extra `d` can remove.
That gap between the two curves is the entire lesson in one picture.

This is the same capacity story that associative-memory theory has told for
decades. Classical Hopfield networks store about `0.14·d` patterns before recall
breaks (Hopfield, 1982). Modern Hopfield networks — which turn out to be
mathematically equivalent to attention — push that capacity up exponentially by
making the read *nonlinear* (Ramsauer et al., 2021). Our linear memory sits at
the conservative end of that spectrum, and now you can measure exactly where.

You do not have to trust the browser. The README has a ten-line NumPy function
that reproduces the curve; `rms_recall_error(30, 40) ≈ 0.85 ≈ √(29/40)`.

---

## 4. Interference is not the same as decay

There are two different ways a fixed state loses information, and conflating them
is a frequent error.

| Mechanism | Update | Effect | Cost |
|---|---|---|---|
| **Interference** | `Sᵢ = Sᵢ₋₁ + kᵢvᵢᵀ` | all bindings kept with equal weight; they blur as they pile up | old facts buried, never removed; error grows with `N` |
| **Decay** (forget gate) | `Sᵢ = λ·Sᵢ₋₁ + kᵢvᵢᵀ`, `λ < 1` | down-weights the past each step; recent bindings dominate | distant facts genuinely gone, whether or not they still matter |

Gated linear attention, RetNet, Mamba's state updates and DeltaNet all live in
this table — they differ mainly in *how* `λ` and the write are computed (scalar,
diagonal, data-dependent, or a rank-1 correction). The explainer's §4 plots recall
error against binding age: at `λ = 1` the curve is flat (interference hits every
binding about equally); drop `λ` and it tilts (recent sharp, old gone). Neither is
"better." A model answering questions about the start of a long document wants
`λ` near 1 and pays in interference; a model tracking a fast-changing state wants
low `λ` and pays in amnesia.

The **delta rule** (DeltaNet, Yang et al., 2024) is the third option in the
explainer: before writing binding `i`, subtract the memory's current prediction
for `kᵢ`, so you write only the correction `vᵢ − Sᵢ₋₁ᵀkᵢ`. This removes
interference for repeated or highly similar keys, at the cost of a slightly more
complex, less trivially parallel update.

---

## 5. This *is* linear attention

Take softmax attention and replace `exp(qᵀk)` with a plain feature-map dot product
`φ(q)ᵀφ(k)` (Katharopoulos et al., 2020):

```
linear:  outᵢ = Σⱼ≤ᵢ ( φ(qᵢ)·φ(kⱼ) ) vⱼ
              = φ(qᵢ)ᵀ ( Σⱼ≤ᵢ φ(kⱼ) vⱼᵀ )
              = φ(qᵢ)ᵀ Sᵢ
```

The parenthesised term `Sᵢ = Σⱼ≤ᵢ φ(kⱼ)vⱼᵀ` is our memory, and it obeys
`Sᵢ = Sᵢ₋₁ + φ(kᵢ)vᵢᵀ` — the exact Hebbian write from §1. So "linear attention,"
"a fast-weight matrix," and "a Hebbian associative memory" are three names for one
object. Moving from softmax to linear is precisely moving from *a cache you scan*
to *a state you accumulate*.

The explainer's §5 puts six bindings under a softmax read and a linear read side
by side. Softmax can concentrate almost all its weight on the true match; the
linear read leaves a tail on the other bindings — that tail is the cross-talk.
Switch the feature map `φ` to `relu` and the contributions become non-negative
and sparse: most bindings contribute exactly zero. That is the doorway to BDH.

---

## 6. Where this lives: Dragon Hatchling (BDH) and BDH-CQ

Pathway's Dragon Hatchling is a Post-Transformer architecture whose stated design
move is to *reformulate attention as synaptic memory that updates as the model
reads* (Kosowski et al., 2025). The concept in this explainer is not an analogy
for BDH's mechanism — the fast-weight / linear-attention state **is** the
mechanism, in BDH's GPU-friendly form.

**The shared skeleton.** BDH-GPU generates its keys and values from
**ReLU-low-rank** transformations of a **sparse, non-negative** activation vector
(around 5% of neurons active in reported runs, with sparsity varying with
predictability), then runs the linear-attention update above over a synapse-level
state `σ` that is shared across memory, adaptation and reasoning. The outer-product
write in §1 is that Hebbian rule; BDH makes it trainable and scalable.

**What BDH adds, with evidence labels:**

| Ingredient | Role | Evidence level |
|---|---|---|
| ReLU sparse non-negative activations (~5%) | keeps "keys" concept-selective, so synaptic writes stay legible and interference stays low | developer-reported |
| Linear attention over ReLU-low-rank features (BDH-GPU) | the GPU-friendly equivalent of the neuron–synapse dynamics — the write/read you explored | formal, in paper |
| Monosemantic synapses; scale-free connectivity | individual synapses reported to encode single concepts; a few hub neurons carry many connections | developer-reported interpretability observations |
| Pretraining scaling 1B → ~600B params; SageMaker HyperPod / AWS integration | evidence the formulation trains at scale on standard infrastructure | developer-reported early experiments |
| Sudoku-Extreme result (BDH) | constraint-reasoning benchmark cited as evidence the evolving state supports multi-step inference | developer-reported benchmark, not an independent reproduction |

**BDH-CQ: the additive-per-demonstration case.** BDH-CQ is a later system that
learns from demonstrations and reasons without a written chain of thought. Its
technical report relates its contextual memory to attention, fast-weight memory
and linear attention, and names a special case where the state **accumulates
additively, one contribution per demonstration** — exactly `S = Σ kᵢvᵢᵀ` with `i`
indexing demonstrations instead of tokens. Two consequences follow directly:

- **Adaptation without weight updates.** BDH-CQ reports solving unseen
  ARC-AGI-style tasks with no evaluation-task demonstrations in training and no
  parameter updates at inference. In our terms, the "learning" is the accumulation
  of outer products into `S` during the forward pass — the weights are frozen, the
  *state* carries the new rule. Contrast HRM and TRM, which take a backward pass
  per task (augmentation into training samples, a learned puzzle embedding, voting
  over augmentations).
- **Demonstration coverage is an interference-budget question.** More
  demonstrations sharpen the recalled rule, up to the point where their key
  overlap starts blurring it — the same trade-off as §3. BDH-CQ's low / medium /
  high "effort" settings spend more *recurrent* compute reading that state, not
  more chain-of-thought tokens.

**Where BDH-CQ has no direct role.** The `√(N/d)` capacity law and the
softmax-vs-linear comparison are properties of linear attention in general.
BDH-CQ neither introduced nor depends on them; it inherits them by building on the
family. Saying otherwise would be inventing a connection.

**Evidence discipline.** Everything labelled *developer-reported* above comes from
Pathway's own paper or technical report, not from an independent reproduction. A
benchmark score is not a deployment; a reported number is not an external
replication. The toy in the explainer is an illustration — a hand-built memory,
not a BDH checkpoint. What *is* solid: the linear-attention and associative-memory
results in §§1–5 are standard, published and widely replicated.

---

## 7. Limitations, stated plainly

1. **The toy uses near-ideal keys.** We hand it random or deliberately correlated
   unit vectors. A real model must *learn* to place near-orthogonal keys, and
   often does not — so real linear-attention models forget more than the `√` law's
   best case.
2. **Linear attention is strictly less expressive than softmax + full cache** for
   exact retrieval. A growing KV cache can return any past value exactly; a fixed
   state cannot, once you exceed its rank. Benchmarks that reward precise
   long-range copying expose this.
3. **Within-session memory is not learning.** Everything here happens in `S`
   during one forward pass. Close the session and it is gone. Consolidating useful
   fast state into durable slow weights is an open problem the BDH authors name
   explicitly.
4. **The `√` law is an average.** It assumes random keys and unit values.
   Structured or adversarial keys, or heavy-tailed value norms, break it in both
   directions.
5. **Illustration vs computation.** §§1–5 of the explainer are live arithmetic in
   your browser. §6's BDH equations are transcribed and simplified from the
   papers; the neuron–synapse framing is a teaching schematic, not a trace of a
   real BDH run.

---

## References

Every claim in the explainer is cited beside where it is used. Primary sources:

- J. Schmidhuber (1992). *Learning to control fast-weight memories.* Neural Computation 4(1).
- J. Ba, G. Hinton, V. Mnih, J. Leibo, C. Ionescu (2016). *Using Fast Weights to Attend to the Recent Past.* NeurIPS 2016.
- J. Hopfield (1982). *Neural networks and physical systems with emergent collective computational abilities.* PNAS 79(8).
- H. Ramsauer et al. (2021). *Hopfield Networks is All You Need.* ICLR 2021.
- A. Katharopoulos, A. Vyas, N. Pappas, F. Fleuret (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* ICML 2020.
- S. Yang, B. Wang, Y. Shen, R. Panda, Y. Kim (2024). *Gated Linear Attention Transformers with Hardware-Efficient Training.* ICML 2024.
- S. Yang, B. Wang, Y. Zhang, Y. Shen, Y. Kim (2024). *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* NeurIPS 2024.
- A. Kosowski et al. (2025). *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain.* Pathway. arXiv:2509.26507. With the Pathway posts "From Attention to Synapses: Deriving BDH", "Why BDH Uses a Brain-Inspired Architecture", "The Equations of Reasoning", and the official toy BDH implementation.
- Pathway (2025–2026). *BDH-CQ technical report.*
- HRM / TRM on ARC — cited as the optimization-route contrast.

**Every BDH and BDH-CQ figure in this post must be verified against
arXiv:2509.26507 and the BDH-CQ technical report before submission.** Points that
depend on a specific unreproduced number are flagged in the repository.

---

*Built as a single self-contained HTML file. Code MIT, text CC-BY-4.0. AI
assistance disclosed in `DISCLOSURE.md`. The team understands and can defend every
equation, control and citation here.*
