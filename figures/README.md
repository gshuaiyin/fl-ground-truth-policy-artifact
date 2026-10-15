# Figure exports and accessible data

The `v3.5.0/` directory contains the 11 permitted final graphic sets aligned with JSS LaTeX v2.1.0 (SVG, PDF, PNG) and their underlying author-generated numerical tables. `supporting_data/` contains additional derived tables. Graphic numbering matches the study; Figure 2 is withheld because it contains upstream source excerpts.

| Figure | Description and data interpretation |
|---|---|
| 1 | Fixed-ranking analysis workflow; distinct evidence populations, not a sample funnel or causal diagram |
| 3 | Java evaluability counts and paired MRR contrasts; error-level paired-bootstrap 95% intervals |
| 4 | Exact target-count subset expectations and per-fault Top-10 changes for 112 faults |
| 5 | Class-only versus class-and-failure controls on the same 206 faults |
| 6 | Python paired changes for all 120 common faults; matched random control for 28 faults; randomization ranges differ from confidence intervals |
| 7 | Separate Transfer-FL effects and paired-bootstrap intervals for each of five seeds; each reuses the same 240 faults |
| S1 | Potentially overlapping analysis populations; counts cannot be summed |
| S2 | Project sensitivity and leave-one-project-out estimates |
| S3 | Derived counts of original versus frozen AST target sets; not original human-review forms |
| S4 | Joint reciprocal-rank and GMAR-A changes; marker area counts overlapping coordinates and origin counts are reported explicitly |
| S5 | Python primary-to-expanded policy effects |

Use the CSV tables as accessible numerical alternatives. Public replay does not generate or redistribute Figure 2. It writes to a separate directory and keeps all frozen inputs unchanged.

The replay uses an opaque white background, 300-DPI PNG previews, PDF/SVG vector exports, and DejaVu Sans. The original files may reference other locally available fonts. A successful export check is not a journal or accessibility certification.

Figures 1 and S1 use the final English submission exports. Figure 1 includes its native draw.io source; S1 has editable SVG text. Public replay copies these two schematics and regenerates nine statistical panels. See [editing and replay scope](../docs/FIGURE_EDITING.md).
