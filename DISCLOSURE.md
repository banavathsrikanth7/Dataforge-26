# AI assistance, code, data, asset and license disclosure

## AI assistance

The initial scaffold of this submission was produced with AI assistance
(**Claude, Anthropic**), used as a coding and drafting tool. Specifically, AI was
used to:

- draft the structure and JavaScript of `index.html` (the interactive explainer),
- draft `README.md`, `one-page-summary.md`, and `blog.md`,
- derive and cross-check the `√(N/d)` recall-error law against a live Monte-Carlo
  simulation and an independent NumPy reimplementation.

AI was **not** used to invent citations, benchmark numbers, or architectural
claims. Every reference points to a real, locatable paper or Pathway publication.
Claims specific to BDH / BDH-CQ are attributed to the primary paper
(arXiv:2509.26507) and the BDH-CQ technical report and are labelled by evidence
level; any that rest on a specific figure carry an explicit "verify against
primary source" marker in the README and in §6 of the explainer.

The registered team is responsible for understanding and defending every
component: the equations, the JavaScript, every control's mapping to a concept
variable, and every citation.

## Code

| Component | Origin | License |
|---|---|---|
| `index.html` — HTML, CSS, all JavaScript (math core, drawing, panels) | original, written for this submission | MIT |
| `mulberry32` pseudo-random generator | well-known public-domain algorithm (attributed to Tommy Ettinger; popularised by `bryc`), reimplemented from the standard form | public domain |
| Box–Muller `randn`, `mulberry32` usage, matrix helpers | original | MIT |

No frameworks, no bundlers, no external scripts, no CDN references. The file runs
offline.

## Data

- **No datasets** are used, bundled, or downloaded.
- All keys and values in the explainer are synthetic random vectors from a seeded
  RNG, generated at view time.

## Model weights

- **None.** There is no trained model in the artifact. It is a hand-built
  associative memory used to illustrate a mechanism.
- No BDH or BDH-CQ checkpoint is used, required, or claimed.

## Assets

| Asset type | Detail |
|---|---|
| Fonts | System font stack only (`-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif` and the platform monospace stack). No font files bundled or fetched. |
| Images | None. All charts and the heatmap are drawn at runtime on `<canvas>`. |
| Icons | One Unicode character (`◐`) for the theme toggle. |
| Audio / video | None. |

## Text and citations

- Explainer prose, `one-page-summary.md`, and `blog.md` are original.
- At most one short quoted phrase per response/section, in quotation marks with
  attribution, per the track's copyright rule. No figures or long passages are
  reproduced from any source.
- Reference list is in `index.html` (§ References), `README.md`, and `blog.md`.

## Licenses

- **Code:** MIT (see `LICENSE`).
- **Text and figures:** CC-BY-4.0.
- **Cited works:** remain under their publishers' terms; only bibliographic data
  and short attributed phrases are used here.

## Forks

This submission is **not** a fork. All files are original to this project.
