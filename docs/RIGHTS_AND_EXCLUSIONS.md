# Publication boundary

Review date: 2026-10-11. This is a file-selection record, not a guarantee of legal clearance.

The owner authorized a public repository while retaining copyright, and explicitly prohibited uploading the manuscript and original human-review materials. A new clean Git root is used so earlier supplementary prose cannot become public through repository history. The previous repository and its history remain private under a separate archive name.

## Included

- Author-written statistical analysis and plotting code, with synthetic test fixtures.
- Author-generated estimates, confidence intervals, randomization summaries, derived per-fault numerical measurements, and checksum identities.
- Original diagrams and statistical figures without source excerpts.
- Derived AST target-count summaries and aggregate review-analysis statistics. No original forms, written reviewer comments, or normalized individual review decisions are included.
- Upstream URLs, recorded versions, and provenance metadata.

Upstream location arrays and certain detailed reference records are removed from the frozen JSON projection. [RESULT_PROJECTION.json](RESULT_PROJECTION.json) records these omissions; they do not change the retained statistical estimates.

## Excluded

| Material | Reason |
|---|---|
| Main manuscript and supplementary-methods prose | Owner's explicit exclusion |
| Original A/B review forms, returns, evidence pages, and normalized individual decisions | Owner's exclusion; evidence also contains upstream source excerpts |
| Current Figure 2 and its source-containing generation function | Contains upstream source-code excerpts |
| Raw Defects4J spectra, annotation corpus, source checkouts, and patches | Not republished; referenced upstream |
| FauxPy code, packaged raw results and source inputs | Upstream GPL material is acquired separately; no GPL code is relicensed here |
| LLMAO code, raw upstream outputs and model-input dataset | No license file found in the inspected version or current upstream tree |
| TRANSFER code, features, full rankings and weights | No license file found in the inspected version or current upstream tree; no redistribution grant established |
| Transfer training/validation runner implementations from the private working tree | Conservatively withheld because of their relationship to the unlicensed upstream implementation |
| Credentials, server addresses, local environment/cache files | Outside the public file allowlist |

Current upstream checks found Defects4J's MIT license and FauxPy experiments' GPL-3.0 license. LLMAO and TRANSFER returned no repository-license file and their recursive current trees contained no license/copying file. Public readability alone is not a redistribution grant. See [GitHub's licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).

No additional license is applied to the original files in this repository. See [../COPYRIGHT.md](../COPYRIGHT.md).
