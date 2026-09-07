# The Interference Budget: fast weights, linear attention, and the synaptic memory of BDH

**Concept.** A *fast-weight* memory replaces the growing key–value cache of softmax
attention with one fixed matrix `S`, updated by Hebbian outer products as tokens
arrive: `Sᵢ = Sᵢ₋₁ + kᵢvᵢᵀ`, read as `out = Sᵀq`. Seen as a running state, this *is*
linear attention (Katharopoulos et al., 2020), and it is the mechanism Pathway's
**Dragon Hatchling (BDH)** uses when it "reformulates attention as synaptic memory
that updates as the model reads" (Kosowski et al., 2025).

**Central claim.** Because every binding is summed into the same matrix, reading
back an earlier value returns the true value **plus a cross-talk term**
`Σⱼ≠ᵢ (kᵢ·kⱼ)vⱼ`. For random unit keys in `d` dimensions the expected recall error
scales as **√(N/d)** for `N` stored bindings; it is near-zero only when keys are
near-orthogonal, and average key correlation `ρ` adds a floor `√((N−1)ρ²)` that
more dimensions cannot remove. A fixed state can therefore *process* an unbounded
stream but cannot *retain* unbounded content: forgetting through interference is
forced by linear algebra, not a tunable defect.

**Design pressure and trade-off.** Softmax attention keeps every past token in a
KV cache, so memory and per-token compute grow with sequence length. Fast weights
trade that cache for constant memory and linear-time inference. The cost is
expressivity: a rank-`min(d,m)` state cannot reproduce arbitrary long-range
retrieval the way a full cache can, which shows up on precise long-copy benchmarks
(Yang et al., 2024).

| Approach | State | Cost/token | Exact recall | Adaptation |
|---|---|---|---|---|
| Softmax + KV cache | grows O(t) | O(t) | yes, to context limit | in-context, lossless |
| Linear attention / fast weights (2020) | fixed d×m | O(1) | lossy, √(N/d) error | additive in state |
| Gated LA / DeltaNet (2024) | fixed d×m | O(1) | better: gating + error-correcting write | gated in state |
| **BDH-GPU** (2025) | synaptic σ over ReLU-low-rank features | O(1) | lossy, aided by ~5% sparse activations | Hebbian in synaptic state |

A decay `λ<1` buys recency by genuinely losing old information; the delta rule
subtracts the current prediction before writing, cutting interference for repeated
keys. RetNet, Mamba's state updates, GLA and Gated DeltaNet differ mainly along
these two axes.

**Where BDH and BDH-CQ sit.** BDH-GPU builds its keys and values from
**ReLU-low-rank** transformations of a sparse, non-negative activation vector
(~5% of neurons active in reported runs, varying with predictability), then runs
the linear-attention update over a synapse-level state σ shared across memory,
adaptation and reasoning (Kosowski et al., 2025; "From Attention to Synapses").
Sparsity keeps the "keys" concept-selective, which the authors link to reported
monosemantic synapses and scale-free connectivity — developer-reported
interpretability observations, not independent reproductions. **BDH-CQ**, a later
system that learns from demonstrations and reasons without a written chain of
thought, relates its contextual memory to attention, fast-weight memory and linear
attention, and names a **special case where state accumulates additively, one
contribution per demonstration** — exactly `S = Σ kᵢvᵢᵀ` with `i` indexing
demonstrations. BDH-CQ reports solving unseen ARC-AGI-style tasks with no
evaluation-task demonstrations in training and no inference-time parameter updates:
the "learning" is the accumulation of outer products during the forward pass, in
contrast to HRM/TRM, which take a per-task backward pass. Its low/medium/high
"effort" settings spend more *recurrent* compute reading that state, not more
chain-of-thought tokens.

**Evidence, labelled.** The linear-attention and associative-memory results — the
√(N/d) capacity law, the cross-talk decomposition, the delta rule — are external
and widely replicated; classical Hopfield capacity is ≈ 0.14·d (Hopfield, 1982),
raised by a nonlinear read (Ramsauer et al., 2021). Everything specific to BDH —
~5% sparsity, Sudoku-Extreme reasoning, 1B→~600B pretraining-scaling experiments,
SageMaker HyperPod integration, BDH-CQ's ARC-AGI numbers — is reported by Pathway
in its paper and technical report. A benchmark is not a deployment; a
developer-reported number is not an external reproduction; the interactive toy
accompanying this summary is a hand-built memory, not a BDH checkpoint.

**Most important limitation.** The √(N/d) law assumes the model has *learned* to
place near-orthogonal keys; real linear-attention models often do not, so they
forget more than the best case. And none of this is durable learning — the state
lives for one session, and consolidating useful fast state into slow weights is an
open problem the BDH authors name explicitly. Where BDH-CQ has no direct role — the
capacity law itself — this summary says so rather than inventing a connection.

**Read next:** Katharopoulos et al., *Transformers are RNNs* (ICML 2020); Yang et
al., *Gated Linear Attention* (ICML 2024) and *DeltaNet* (NeurIPS 2024); Kosowski
et al., *The Dragon Hatchling* (arXiv:2509.26507, 2025) and the BDH-CQ technical
report; Ramsauer et al., *Hopfield Networks is All You Need* (ICLR 2021).
