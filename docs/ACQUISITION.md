# Separately acquired inputs

The following third-party inputs are referenced, not mirrored. Check each upstream project's terms before downloading, using, or redistributing it. These instructions identify inputs and conventions; they are not a tested end-to-end reproduction claim.

| Resource | Recorded study version | Upstream |
|---|---|---|
| Defects4J | v1.2 | https://github.com/rjust/defects4j |
| Pearson et al. annotation and coverage artifacts | Local frozen annotation archive identifier `db9a70bfeaef` | https://fault-localization.cs.washington.edu/ and its linked Bitbucket repository |
| FauxPy experiments / BugsInPy | `fd2d8fddce63d2e4ff93150aeff7f9e17d004657` | https://github.com/atom-sw/fauxpy-experiments |
| LLMAO | `60f9e5c7dfaab920aed22a33b596299c5ce3c429` | https://github.com/squaresLab/LLMAO |
| TRANSFER | Study acquisition commit was not recorded | https://github.com/mxx1219/TRANSFER |

The inspected current TRANSFER master is `c5ff7bd8420a08afb23f44ad5ea3ffe8337d4fdd`; this identifies the 2026-10-11 rights-review snapshot, **not** the unrecorded original study acquisition commit. Do not treat it as proof of byte identity with the study input.

## Java inputs

The coverage host documents per-fault files under `data/<Project>/<Bug>/gzoltar-files.tar.gz`. The archive contains `matrix` and `spectra`. Obtain patches from a separately acquired Defects4J version and annotations from the coverage site's linked repository. Analysis modules expect the private-input conventions under `data/` and `external/`; those directories are deliberately ignored by Git.

See command-line help for the author-written analyses:

```bash
python src/revision_v140.py --help
python src/counterfactual_multiplicity.py --help
python src/bugsinpy_external_validation.py --help
python src/llmao_external_validation.py --help
python scripts/analyze_transfer_full_identity_multiseed.py --help
```

When running these commands from a checkout, the scripts and their sibling modules resolve imports from `src/` normally. Obtain all required external inputs first; the public frozen replay does not require them.

## Transfer-FL

The upstream README identifies its Google Drive dataset and expects files such as `dataset_fl.pkl`, `src_code.pkl`, and derived semantic/spectrum/mutation features. Use that project's own instructions for acquisition and training. No upstream training implementation or feature archive is distributed here.

The study ran seeds `888`, `1888`, `2888`, `3888`, and `4888` and preserved full `class@line` identities. `SHA256SUMS.rankings.txt` records 1,975 private ranking identities. The present public package contains downstream numerical analysis, but no complete rank files or the 50 fold weights. Their redistribution status is unresolved. No public download endpoint is claimed for them.

## Human review and provenance limits

The original forms, reviewer returns and evidence pages remain private by the owner's instruction. Public aggregate comparisons and derived AST count tables permit statistical inspection but do not recreate the independent review process. LLMAO's exact model-input source snapshot remains unavailable. These limits must remain explicit in any data-availability statement.
