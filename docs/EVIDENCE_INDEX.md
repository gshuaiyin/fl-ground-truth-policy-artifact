# Evidence map

| Evidence family | Public frozen file |
|---|---|
| Revised Java protocol and matched controls | `results/revision_v140/run_20260910_r3/analysis.json` |
| Historical Java sensitivity | `results/full/groundtruth_sensitivity.json` |
| Target-count counterfactual | `results/full/counterfactual_multiplicity.json` |
| Policy decomposition | `results/full/policy_decomposition.json` |
| Project sensitivity | `results/full/project_robustness.json` |
| Calibration and tie sensitivity | `results/full/gmar_null_calibration.json`, `gmar_tie_sensitivity.json` |
| Class-matched control | `results/full/class_matched_placebo.json` |
| Python validation | `results/full/bugsinpy_external_validation.json` |
| Derived audit counts | `results/manual_audit_summary/target_counts.json` |
| Aggregate review-analysis statistics | `results/manual_audit_36/reviewer_b_full_analysis.json` (individual rows omitted) |
| LLMAO re-evaluation | `results/llmao_external_validation/analysis.json` (upstream target lists omitted) |
| Transfer-FL five-seed analysis | `results/rq5_transfer_repair/outputs/transfer_full_identity_multiseed_analysis_v2/analysis.json` (location lists omitted) |
| Official fold consistency check | `results/rq5_transfer_repair/outputs/validation/official_fold1.json` (per-fault upstream reference ranks omitted) |

The 11 original graphic sets and their data tables are in `figures/v3.5.0/`; the derived supplementary tables are in `figures/supporting_data/`. Figure numbering preserves the study: the missing Figure 2 is intentional. See [FIGURE_PROVENANCE.json](FIGURE_PROVENANCE.json).

The five Transfer-FL seeds share a benchmark and do not constitute five independent datasets. The fold consistency check is not a full paper reproduction. The derived AST counts are not original human-review decisions. Full experiment input acquisition and end-to-end reproduction are outside the verified public replay scope.
