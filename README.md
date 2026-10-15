# Ground-truth policy sensitivity in fault localization

Public code and statistical replay package aligned with Chinese manuscript v3.5.0, English v2.1.0, and JSS LaTeX v2.1.0. Scientific estimates remain those of the frozen v3.4.1 study evidence. This package provides author-written analysis code, synthetic tests, frozen numerical results, 11 original figures, figure data tables, and a portable figure replay script.

**Copyright is retained. No additional open-source or data redistribution license is granted.** Upstream materials retain their own terms. See [COPYRIGHT.md](COPYRIGHT.md).

The manuscript, supplementary-methods prose, original human-review forms/returns, individual normalized reviewer decisions, and evidence pages are excluded. Third-party source code, raw spectra/features, full Transfer-FL rankings, model weights, and the source-containing Figure 2 are also excluded. The statistical count summaries of the audit are derived outputs, not the original review records.

## Verify the public package

Use Python 3.13. Create an environment and install the dependencies:

```bash
python -m venv .venv
```

Activate the environment before installing: `source .venv/bin/activate` on Linux/macOS, or `.venv\Scripts\Activate.ps1` in Windows PowerShell.

```bash
python -m pip install -r environment/requirements-core.txt
```

Linux/macOS:

```bash
PYTHONPATH=src python -B -m unittest discover -s tests -p 'test_*.py'
python -B scripts/verify_release.py
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = 'src'
python -B -m unittest discover -s tests -p 'test_*.py'
python -B scripts/verify_release.py
```

## Replay the figure values

```bash
python -m pip install -r environment/requirements-figures.txt
python -B scripts/build_public_figures.py
python -B scripts/check_figure_tables.py
```

Replay exports go to `replay/figures/` and do not overwrite the original SVG/PDF/PNG files in `figures/v3.5.0/`. The replay regenerates nine statistical panels from frozen results and copies the two approved authored schematics (Figures 1 and S1). It preserves estimates, paired samples, and uncertainty definitions. Fonts and file metadata of regenerated statistical panels can differ. Editable diagram sources and their scope are documented in [docs/FIGURE_EDITING.md](docs/FIGURE_EDITING.md). No experiments are retrained and no bootstrap intervals are re-estimated.

Figure 2 is deliberately absent because it contains upstream source excerpts. Its absence is documented, rather than substituted with a fabricated example.

## Contents

| Directory | Contents |
|---|---|
| `src/` | Original policy, ranking-metric, control, and robustness code |
| `tests/` | Unit tests with author-written synthetic fixtures |
| `results/` | Frozen statistics; upstream location lists and individual reviewer decisions removed |
| `figures/v3.5.0/` | 6 main figures and 5 supplementary figures, in SVG/PDF/PNG, with statistical tables |
| `figures/supporting_data/` | Additional derived statistical tables |
| `scripts/` | Public replay, table comparison, integrity, and author-written ranking-analysis utilities |
| `docs/` | Evidence map, upstream acquisition information, exclusions, and provenance |

This is a public statistical replay package. It does **not** claim that all raw experimental inputs or a complete independent end-to-end rerun are provided. The hash inventory for 1,975 withheld rankings verifies their expected identities; it is not a download of the rankings. See [docs/ACQUISITION.md](docs/ACQUISITION.md) and [docs/RIGHTS_AND_EXCLUSIONS.md](docs/RIGHTS_AND_EXCLUSIONS.md).

## Reference

Current release: [v3.5.0-public-20261015](https://github.com/gshuaiyin/fl-ground-truth-policy-artifact/releases/tag/v3.5.0-public-20261015). This release has no new Zenodo DOI assigned. The earlier frozen numerical baseline is archived at [10.5281/zenodo.23296795](https://doi.org/10.5281/zenodo.23296795); that DOI identifies the older package, not a byte-identical copy of this release. See [CITATION.md](CITATION.md) and [release changes](docs/RELEASE_CHANGES.md).
