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

**Where BDH and BDH-CQ sit.** The Dragon Hatchling paper (arXiv:2509.26507, §6.1)
defines BDH-GPU as "a specific kind of ReLU-lowrank feed-forward network, and a
linear attention mechanism"; its positive activations are sparse at "about 5%
level" (§6.4), and it matches GPT-2 on language and translation at equal parameters
from 10M to 1B (§4.2). Pathway separately reports monosemantic synapses, a
scale-free heavy-tailed neuron graph, 97.4% on Sudoku Extreme with no chain of
thought (research post, Mar 2026), and pretraining scaling laws holding to ~600B
parameters (largest *benchmarked* model in the family is 150M). **BDH-CQ**
(arXiv:2608.09888) adds demonstration-driven latent reasoning: its state update
`Sₜ = Uθ(Sₜ₋₁, Dₜ)` is described as "generally related to attention, fast-weight
memory, and linear-attention views of contextual association … capturing the
special case `Sₜ = Sₜ₋₁ + Uθ(Dₜ)`" (§3.2) — exactly `S = Σ kᵢvᵢᵀ` with `i` indexing
demonstrations. "Neither task identifiers nor evaluation-task demonstration pairs
participate in training, and no parameters are updated at inference time"; a 150M
model reaches 29.5% pass@2 on public ARC-AGI-1 at ~$0.0007/task. Contrast HRM/TRM,
whose ARC pipeline is transductive and needs a per-task backward pass (§8). Its
low/medium/high "effort" levels (pass@2 21.0 / 27.0 / 29.5%) spend more *recurrent*
compute, not more chain-of-thought tokens.

**Evidence, labelled.** The linear-attention and associative-memory results — the
√(N/d) capacity law, the cross-talk decomposition, the delta rule — are external
and widely replicated; classical Hopfield capacity is ≈ 0.14·d (Hopfield, 1982),
raised by a nonlinear read (Ramsauer et al., 2021). The ReLU-lowrank + linear
attention formulation and the 10M–1B GPT-2 parity are *formal / in the paper*.
Everything else specific to BDH — the ~5% sparsity measurement, monosemanticity,
Sudoku 97.4%, ARC-AGI 29.5%, the 600B scaling figure, SageMaker HyperPod
development — is *developer-reported* by Pathway or AWS, with no independent
reproduction. A benchmark is not a deployment; the interactive toy accompanying
this summary is a hand-built memory, not a BDH checkpoint.

**Most important limitation.** The √(N/d) law assumes the model has *learned* to
place near-orthogonal keys; real linear-attention models often do not, so they
forget more than the best case. And none of this is durable learning — the state
lives for one session, and consolidating useful fast state into slow weights is an
open problem the BDH authors name explicitly. Where BDH-CQ has no direct role — the
capacity law itself — this summary says so rather than inventing a connection.

**Read next:** Katharopoulos et al., *Transformers are RNNs* (ICML 2020); Yang et
al., *Gated Linear Attention* (ICML 2024) and *DeltaNet* (NeurIPS 2024); Kosowski
et al., *The Dragon Hatchling* (arXiv:2509.26507, 2025) and *BDH-CQ* (arXiv:2608.09888,
2026); Ramsauer et al., *Hopfield Networks is All You Need* (ICLR 2021).
