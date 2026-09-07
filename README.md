# The Interference Budget

**An interactive explainer for DataForge 2026 — Pathway Track.**

> A fixed-size fast-weight memory `S = Σ kᵢvᵢᵀ` stores a stream of key→value
> bindings **without ever growing**, but because every binding is added into the
> **same matrix**, reading back an earlier value returns the true value **plus a
> cross-talk term** `Σⱼ≠ᵢ (kᵢ·kⱼ)vⱼ` whose size is set by how much the keys
> overlap — so recall error grows like **√(N/d)** and is near-zero only when the
> keys are near-orthogonal.

That sentence is the one falsifiable claim the whole artifact teaches. Every
control exists so a learner can try to break it.

---

## Links

| Item | URL |
|---|---|
| Public artifact (opens without sign-in) | https://dataforgepathway.netlify.app/ |
| Source repository | https://github.com/banavathsrikanth7/Dataforge-26|
| One-page concept summary (PDF) | [`one-page-summary.pdf`](one-page-summary.pdf) |
| Blog (PDF) | [`blog.pdf`](blog.pdf) |

---

## The claim, the audience, the objectives

**Claim (falsifiable).** Recall error of the additive fast-weight memory
`S = Σ kᵢvᵢᵀ` grows as `√(N/d)` for random unit keys and is near-zero only for
near-orthogonal keys; correlated keys add a floor that more dimensions cannot
remove. If a learner can keep recall sharp while packing in many correlated keys,
the claim is wrong.

**Intended learner.** An ML-literate student or early practitioner who has seen
softmax attention (`softmax(QKᵀ)V`), can read a matrix–vector product, and has
trained something small in PyTorch or NumPy.

**Prerequisites.** Dot products, outer products, vector norms; the idea that
softmax attention keeps one key–value pair per past token in a cache that grows
with sequence length.

**Learning objectives.** After using the artifact a learner can:

1. Write the fast-weight update and read as two one-line equations and name every symbol.
2. Predict, before running it, whether a given `(N, d, key-correlation)` setting recalls cleanly.
3. Derive the cross-talk term and explain why it is exactly zero for orthogonal keys.
4. Show that this memory *is* linear attention with a running state, not an approximation bolted on.
5. Point to where the mechanism sits inside Dragon Hatchling (BDH) and BDH-CQ, and name what BDH adds.
6. State the key limitation: a fixed state forgets through interference, and within-session memory is not durable learning.

**Sixty-second test.** Sandbox at `d = 48, ρ = 0, N = 6` (recall error ≈ 0.3).
Predict the error after dragging `N` to 24 without changing `d` (√4 = 2×, so
≈ 0.6–0.7), then check against the number the sandbox prints and explain why in one
sentence.

---

## Architecture of the artifact

A **single self-contained `index.html`** — no build step, no dependencies, no
network calls, no analytics, no storage. Open the file and it runs.

| Part | What it is | Live / precomputed / illustrative |
|---|---|---|
| Hero demo | 20 bindings stream into a 64×10 memory; fact #2 is probed as the stream grows | **Live** JS: real `buildState` / `readState` each frame |
| §1 The write | Step through 20 Hebbian outer-product writes, watch `S` fill in | **Live** JS |
| §2 The read | Probe a stored key; decompose `Sᵀkᵢ` into signal + cross-talk; truth-vs-estimate bars; delta-rule toggle | **Live** JS |
| §3 The √ law | Sweep `N` or `d`, average many trials, plot measured RMS error against the predicted `√((N−1)(ρ² + (1−ρ²)/d))` curve | **Live** JS (Monte-Carlo in browser) |
| §4 Forgetting | Recall error vs binding age under decay `λ`; interference (flat) vs decay (tilted) | **Live** JS |
| §5 = Linear attention | Same six bindings under softmax vs linear reads; feature-map `φ` toggle (identity / elu+1 / relu) | **Live** JS |
| §6 Inside BDH | Shared-skeleton equations, "what BDH adds" table with evidence labels, BDH-CQ additive-per-demonstration case | **Transcribed & simplified** from primary sources; the neuron–synapse description is a teaching schematic |
| §7 Limitations | Five honest caveats and failure cases | Prose |
| Sandbox | Every control (`N, d, m, ρ, λ, probe, delta rule, unit-norm values`) in one panel with a verdict | **Live** JS |

### Role of every major component (source)

- `mulberry32`, `randn` — deterministic RNG (per-panel seed) so every result reproduces.
- `makeKeys(N, d, ρ)` — `N` unit vectors with target pairwise correlation `ρ`, built as `normalize(√(1−ρ)·gᵢ + √ρ·base)`.
- `makeValues(N, m)` — random vectors, unit-normalised by default so `‖v‖ = 1` and recall error equals `‖cross-talk‖` exactly.
- `buildState(K, V, λ, delta)` — the memory: `Sᵢ = λSᵢ₋₁ + kᵢwᵢᵀ`, with `wᵢ = vᵢ` (Hebbian) or `wᵢ = vᵢ − Sᵢ₋₁ᵀkᵢ` (delta rule, DeltaNet with β = 1).
- `readState(S, q)` — `Sᵀq = Σⱼ (kⱼ·q) vⱼ`.
- `crosstalkVec(K, V, p)` — the analytic `Σⱼ≠ₚ (kₚ·kⱼ) vⱼ`, drawn beside the measured error.
- `predictedError(N, d, ρ)` — `√((N−1)(ρ² + (1−ρ²)/d))`, the theoretical RMS recall error.

---

## How to reproduce the results

Everything is in the browser; there is nothing to install.

```bash
# Option A — just open it
open index.html          # macOS
start index.html         # Windows
xdg-open index.html      # Linux

# Option B — serve locally (identical behaviour)
python -m http.server 8000
# then visit http://localhost:8000
```

- The √-law plot (§3) is a Monte-Carlo average computed live. Raise "trials per
  point" for a smoother curve; the seed schedule (`1000 + t*7 + N*13 + d*101`) is in
  the `measureError` closure so a given curve reproduces exactly.
- To check the capacity law yourself outside the browser, the same recurrence in
  NumPy:

  ```python
  import numpy as np
  def rms_recall_error(N, d, m=10, rho=0.0, trials=200, seed=0):
      rng = np.random.default_rng(seed); errs = []
      for _ in range(trials):
          base = rng.standard_normal(d); base /= np.linalg.norm(base)
          G = rng.standard_normal((N, d))
          K = np.sqrt(1-rho)*(G/np.linalg.norm(G,axis=1,keepdims=True)) + np.sqrt(rho)*base
          K /= np.linalg.norm(K, axis=1, keepdims=True)
          V = rng.standard_normal((N, m)); V /= np.linalg.norm(V, axis=1, keepdims=True)
          S = K.T @ V                       # Σ kᵢ vᵢᵀ
          p = rng.integers(N)
          errs.append(np.linalg.norm(S.T @ K[p] - V[p]))
      return float(np.mean(errs))
  # rms_recall_error(30, 40) ≈ 0.85 ≈ sqrt(29/40)
  ```

---

## Which parts are live, precomputed, synthetic, or animated

- **Live computation:** §§1–5 and the Sandbox — every number and pixel comes from
  `buildState` / `readState` / Monte-Carlo running in your browser at view time.
- **Synthetic data:** all keys and values are random vectors from a seeded RNG.
  There is no trained model anywhere in the artifact.
- **Transcribed / simplified:** §6's BDH and BDH-CQ equations are taken from the
  primary sources and reduced to the shared skeleton; notation is simplified.
- **Illustrative:** the neuron–synapse framing in §6 is a teaching schematic, not a
  trace of a real BDH run. Nothing in the artifact is a scripted animation presented
  as model behaviour.

---

## BDH / BDH-CQ integration

The concept is not analogous to BDH's mechanism — the fast-weight / linear-attention
state **is** the mechanism in BDH-GPU's formulation. §6:

- states plainly that BDH-GPU builds keys/values from **ReLU-low-rank**
  transformations of a **sparse non-negative** activation vector, then runs the
  linear-attention update over a synapse-level state;
- separates what is **formal in the paper** (the linear-attention formulation) from
  what is **developer-reported** (~5% sparsity, monosemantic synapses, scale-free
  connectivity, Sudoku-Extreme, 1B→~600B scaling, HyperPod integration);
- ties BDH-CQ's **additive-per-demonstration** contextual-memory special case
  directly to `S = Σ kᵢvᵢᵀ`, and explains adaptation-without-weight-updates in those
  terms, contrasting HRM/TRM's per-task backward pass;
- explicitly notes where BDH-CQ has **no** direct role (the √(N/d) capacity law is a
  property of linear attention in general).

**All BDH/BDH-CQ claims must be verified against the primary paper
(arXiv:2509.26507) and the BDH-CQ technical report before submission.** Placeholders
where a specific figure needs a citation are marked in `one-page-summary.md` and in
§6 of `index.html`.

---

## Three recent primary papers (2022–2026) on this concept

1. **Yang, Wang, Shen, Panda, Kim — *Gated Linear Attention Transformers with
   Hardware-Efficient Training* (ICML 2024).** Adds data-dependent gating to the
   fast-weight state; the gating axis in §4.
2. **Yang, Wang, Zhang, Shen, Kim — *Parallelizing Linear Transformers with the
   Delta Rule over Sequence Length* (NeurIPS 2024).** Replaces the pure Hebbian
   write with an error-correcting (delta) write to cut interference; the delta-rule
   toggle in §2.
3. **Kosowski et al. — *The Dragon Hatchling: The Missing Link between the
   Transformer and Models of the Brain* (arXiv:2509.26507, 2025).** Reformulates
   attention as a Hebbian synaptic state with sparse non-negative activations;
   reports scaling experiments and Sudoku-Extreme reasoning.

Supporting: Katharopoulos et al., *Transformers are RNNs* (ICML 2020); Ramsauer et
al., *Hopfield Networks is All You Need* (ICLR 2021); Schmidhuber (1992) and Ba et
al. (2016) on fast weights.

Full reference list with inline citations is in `index.html` (§ References) and
`one-page-summary.md`.

---

## Deployment

The artifact is one static file. Any of these gives a public URL with no sign-in:

- **GitHub Pages:** push `index.html` to a repo, enable Pages on the default branch,
  root folder. A `.nojekyll` file is included so Pages serves it verbatim.
- **Netlify / Cloudflare Pages / Vercel:** drag-and-drop the folder, or connect the
  repo; no build command, publish directory = repo root.
- **Local:** `python -m http.server` (see above).

Fill the artifact URL into the Links table above and into `blog.pdf` before
submitting.

---

## Source and license record

| Asset | Source | License |
|---|---|---|
| `index.html` (all code, CSS, JS) | original, written for this submission | MIT |
| Explainer text, `one-page-summary.md`, `blog.md` | original | CC-BY-4.0 |
| `mulberry32` PRNG | public-domain algorithm (Tommy Ettinger / bryc), reimplemented | public domain |
| Fonts | system font stack only (`-apple-system, Segoe UI, Roboto, …`) — none bundled | n/a |
| Graphics / charts | drawn at runtime on `<canvas>` — no image files | n/a |
| Libraries | **none** | n/a |
| Reference content | cited papers (see References); quoted at most one short phrase with attribution | as per publishers |

No datasets, no model weights, no third-party media are bundled or downloaded.

---

## AI assistance disclosure

See [`DISCLOSURE.md`](DISCLOSURE.md). Summary: this scaffold — the `index.html`
structure and JavaScript, `README.md`, `one-page-summary.md`, and `blog.md` — was
drafted with AI assistance (Claude, Anthropic). The registered team is responsible
for verifying the mathematics (done: the `√(N/d)` law is checked against live
Monte-Carlo in §3 and against the NumPy snippet above), verifying every BDH/BDH-CQ
claim against primary sources (**pending — see markers**), and being able to explain
and defend every equation, control, and citation in a live review.

---

## License

- Code: [MIT](LICENSE)
- Text and figures: CC-BY-4.0
