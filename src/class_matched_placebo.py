"""Class-matched randomization control for omission-aware ground truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from gmar import gmar_score
from groundtruth_sensitivity_audit import (
    mapped_line_indices,
    official_locations,
    source_line_index,
)
from spectrum import average_ranks_desc, load_gzoltar, suspiciousness


FORMULAS = ("ochiai", "dstar", "tarantula", "barinel")
TOP_KS = (1, 5, 10, 20)
METRICS = ("mrr", "exam", "gmar_f", "gmar_a", "top1", "top5", "top10", "top20")


def stable_seed(base_seed: int, bug_id: str) -> int:
    digest = hashlib.sha256(bug_id.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def sample_class_matched(
    rng: np.random.Generator,
    pools: dict[str, np.ndarray],
    counts: dict[str, int],
    replicates: int,
) -> np.ndarray:
    columns = []
    for class_name in sorted(counts):
        pool = pools[class_name]
        count = counts[class_name]
        if pool.size < count:
            raise ValueError(f"Insufficient pool for {class_name}: {pool.size} < {count}")
        if count == 1:
            columns.append(rng.choice(pool, size=(replicates, 1), replace=True))
        else:
            samples = np.empty((replicates, count), dtype=int)
            for replicate in range(replicates):
                samples[replicate] = rng.choice(pool, size=count, replace=False)
            columns.append(samples)
    return np.concatenate(columns, axis=1)


def tie_inspection_sizes(scores: np.ndarray) -> np.ndarray:
    values, counts = np.unique(scores, return_counts=True)
    at_or_above = np.cumsum(counts[::-1])[::-1]
    lookup = {float(value): int(size) for value, size in zip(values, at_or_above)}
    return np.asarray([lookup[float(score)] for score in scores], dtype=int)


def metric_values(
    scores: np.ndarray,
    observed_indices: np.ndarray,
    fixed_indices: np.ndarray,
    sampled_indices: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    population = scores.size
    target_count = observed_indices.size
    ranks = average_ranks_desc(scores)
    inspected = tie_inspection_sizes(scores)
    single_gmar = (population - inspected) / population

    observed_best_rank = float(np.min(ranks[observed_indices]))
    observed_best_inspected = int(np.min(inspected[observed_indices]))
    observed = {
        "mrr": 1.0 / observed_best_rank,
        "exam": observed_best_rank / population,
        "gmar_f": float(gmar_score(observed_best_inspected, target_count, population)),
        "gmar_a": float(np.mean(single_gmar[observed_indices])),
        **{f"top{k}": float(observed_best_rank <= k) for k in TOP_KS},
    }

    placebo_best_rank = np.min(ranks[sampled_indices], axis=1)
    placebo_best_inspected = np.min(inspected[sampled_indices], axis=1)
    if fixed_indices.size:
        placebo_best_rank = np.minimum(placebo_best_rank, np.min(ranks[fixed_indices]))
        placebo_best_inspected = np.minimum(
            placebo_best_inspected, np.min(inspected[fixed_indices])
        )
        fixed_gmar_sum = float(np.sum(single_gmar[fixed_indices]))
    else:
        fixed_gmar_sum = 0.0

    gmar_f_cache = {
        int(size): gmar_score(int(size), target_count, population)
        for size in np.unique(placebo_best_inspected)
    }
    placebo = {
        "mrr": 1.0 / placebo_best_rank,
        "exam": placebo_best_rank / population,
        "gmar_f": np.asarray(
            [gmar_f_cache[int(size)] for size in placebo_best_inspected], dtype=float
        ),
        "gmar_a": (
            fixed_gmar_sum + np.sum(single_gmar[sampled_indices], axis=1)
        )
        / target_count,
        **{f"top{k}": (placebo_best_rank <= k).astype(float) for k in TOP_KS},
    }
    return observed, placebo


def empty_accumulator(replicates: int) -> dict:
    return {
        "bugs": 0,
        "observed_target_sum": 0,
        "observed": {metric: 0.0 for metric in METRICS},
        "placebo": {metric: np.zeros(replicates, dtype=float) for metric in METRICS},
    }


def add_result(
    accumulator: dict,
    observed: dict[str, float],
    placebo: dict[str, np.ndarray],
    target_count: int,
) -> None:
    accumulator["bugs"] += 1
    accumulator["observed_target_sum"] += target_count
    for metric in METRICS:
        accumulator["observed"][metric] += observed[metric]
        accumulator["placebo"][metric] += placebo[metric]


def summarize_accumulator(accumulator: dict, replicates: int) -> dict:
    bugs = accumulator["bugs"]
    if not bugs:
        return {"bugs": 0}
    output = {
        "bugs": bugs,
        "mean_targets": accumulator["observed_target_sum"] / bugs,
        "metrics": {},
    }
    for metric in METRICS:
        observed = accumulator["observed"][metric] / bugs
        placebo = accumulator["placebo"][metric] / bugs
        if metric == "exam":
            effect = float(np.mean(placebo) - observed)
            extreme = int(np.sum(placebo <= observed))
            effect_definition = "placebo_minus_observed; positive favors official targets"
        else:
            effect = float(observed - np.mean(placebo))
            extreme = int(np.sum(placebo >= observed))
            effect_definition = "observed_minus_placebo; positive favors official targets"
        output["metrics"][metric] = {
            "observed": float(observed),
            "placebo_mean": float(np.mean(placebo)),
            "placebo_95_randomization_interval": [
                float(np.quantile(placebo, 0.025)),
                float(np.quantile(placebo, 0.975)),
            ],
            "effect": effect,
            "effect_definition": effect_definition,
            "one_sided_randomization_p": (extreme + 1) / (replicates + 1),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spectra_root", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--replicates", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replicates < 100:
        raise ValueError("At least 100 randomization replicates are required")

    spectra_dirs = []
    for project_dir in sorted(path for path in args.spectra_root.iterdir() if path.is_dir()):
        for bug_dir in sorted(
            (path for path in project_dir.iterdir() if path.is_dir()),
            key=lambda path: int(path.name),
        ):
            spectra_dirs.append((project_dir.name, bug_dir.name, bug_dir))

    accumulators: dict[str, dict[str, dict]] = defaultdict(dict)
    skip_reasons: Counter[str] = Counter()
    eligible_rows = []
    for project, bug, bug_dir in spectra_dirs:
        bug_id = f"{project}-{bug}"
        direct, omission = official_locations(args.annotations, project, bug)
        if not omission:
            skip_reasons["no_official_omission_candidates"] += 1
            continue
        spectrum = load_gzoltar(bug_dir)
        source_lines, probe_to_line = source_line_index(spectrum.statements)
        direct_indices = mapped_line_indices(source_lines, direct)
        omission_indices = mapped_line_indices(source_lines, omission)
        direct_set = set(map(int, direct_indices))
        omission_only = np.asarray(
            sorted(set(map(int, omission_indices)) - direct_set), dtype=int
        )
        if not omission_only.size:
            skip_reasons["no_mapped_omission_only_target"] += 1
            continue

        counts: Counter[str] = Counter(source_lines[index][0] for index in omission_only)
        official_indices = direct_set | set(map(int, omission_indices))
        pools = {}
        insufficient = False
        for class_name, count in counts.items():
            pool = np.asarray(
                [
                    index
                    for index, location in enumerate(source_lines)
                    if location[0] == class_name and index not in official_indices
                ],
                dtype=int,
            )
            pools[class_name] = pool
            if pool.size < count:
                insufficient = True
        if insufficient:
            skip_reasons["insufficient_same_class_non_target_pool"] += 1
            continue

        rng = np.random.default_rng(stable_seed(args.seed, bug_id))
        sampled_indices = sample_class_matched(
            rng, pools, dict(counts), args.replicates
        )
        observed_indices = np.asarray(
            sorted(direct_set | set(map(int, omission_indices))), dtype=int
        )
        fault_type = "mixed" if direct_indices.size else "pure_omission"
        eligible_rows.append(
            {
                "bug_id": bug_id,
                "project": project,
                "fault_type": fault_type,
                "direct_targets_fixed": int(direct_indices.size),
                "omission_targets_randomized": int(omission_only.size),
                "total_targets": int(observed_indices.size),
                "matched_classes": dict(sorted(counts.items())),
            }
        )
        groups = ("all", f"project:{project}", f"fault_type:{fault_type}")
        for formula in FORMULAS:
            probe_scores = suspiciousness(spectrum.coverage, spectrum.failed, formula)
            line_scores = np.full(len(source_lines), -np.inf)
            np.maximum.at(line_scores, probe_to_line, probe_scores)
            observed, placebo = metric_values(
                line_scores,
                observed_indices,
                direct_indices,
                sampled_indices,
            )
            for group in groups:
                if formula not in accumulators[group]:
                    accumulators[group][formula] = empty_accumulator(args.replicates)
                add_result(
                    accumulators[group][formula],
                    observed,
                    placebo,
                    int(observed_indices.size),
                )

    summaries = {
        group: {
            formula: summarize_accumulator(accumulator, args.replicates)
            for formula, accumulator in formula_results.items()
        }
        for group, formula_results in sorted(accumulators.items())
    }
    all_summary = summaries.get("all", {})
    all_metrics = [
        metric
        for result in all_summary.values()
        for metric in result.get("metrics", {}).values()
    ]
    sanity = {
        "eligible_bug_count_matches_accumulator": len(eligible_rows)
        == next(iter(all_summary.values()))["bugs"]
        if all_summary
        else False,
        "all_formulas_same_bug_count": len(
            {result["bugs"] for result in all_summary.values()}
        )
        == 1,
        "all_summary_values_finite": bool(all_metrics)
        and all(
            np.isfinite(metric[key])
            for metric in all_metrics
            for key in ("observed", "placebo_mean", "effect")
        ),
        "all_randomization_intervals_ordered": bool(all_metrics)
        and all(
            metric["placebo_95_randomization_interval"][0]
            <= metric["placebo_95_randomization_interval"][1]
            for metric in all_metrics
        ),
        "all_p_values_valid": bool(all_metrics)
        and all(
            1 / (args.replicates + 1)
            <= metric["one_sided_randomization_p"]
            <= 1.0
            for metric in all_metrics
        ),
    }
    sanity["passed"] = all(sanity.values())
    if not sanity["passed"]:
        raise RuntimeError(f"Placebo sanity checks failed: {sanity}")

    output = {
        "design": {
            "unit": "source line",
            "replicates": args.replicates,
            "seed": args.seed,
            "negative_control": "keep mapped official direct targets fixed; replace mapped omission-only targets without replacement by non-target spectrum lines from the same outer classes; preserve per-class target counts",
            "inference": "Monte Carlo randomization distribution; intervals are not bootstrap confidence intervals",
        },
        "scope": {
            "spectra_bugs": len(spectra_dirs),
            "eligible_bugs": len(eligible_rows),
            "skip_reasons": dict(sorted(skip_reasons.items())),
        },
        "sanity_checks": sanity,
        "summaries": summaries,
        "eligible_rows": eligible_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "scope": output["scope"],
                "sanity_checks": sanity,
                "all": {
                    formula: {
                        metric: result["metrics"][metric]
                        for metric in ("mrr", "exam", "gmar_f", "gmar_a", "top10")
                    }
                    for formula, result in all_summary.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
