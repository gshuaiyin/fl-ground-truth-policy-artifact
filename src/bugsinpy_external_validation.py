"""Independent BugsInPy validation using the published FauxPy replication data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from class_matched_placebo import (
    METRICS,
    add_result,
    empty_accumulator,
    metric_values,
    sample_class_matched,
    stable_seed,
    summarize_accumulator,
    tie_inspection_sizes,
)
from gmar import gmar_score
from project_robustness_analysis import exact_sign_p, percentile_interval
from spectrum import average_ranks_desc


FORMULAS = ("Ochiai", "DStar", "Tarantula")
TOP_KS = (1, 5, 10, 20)


def holm_adjust(items: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(items, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (total - index)))
        adjusted[name] = running
    return adjusted


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranking_path(root: Path, project: str, bug: str, formula: str) -> Path:
    return root / f"{project}_{bug}_{formula}_Statement.csv"


def load_ranking(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    entities = [row["Entity"].replace("\\", "/") for row in rows]
    scores = np.asarray([float(row["Score"]) for row in rows], dtype=float)
    if len(entities) != len(set(entities)):
        raise ValueError(f"Duplicate entities in {path}")
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"Non-finite scores in {path}")
    return entities, scores


def target_sets(module_items: list[dict]) -> tuple[set[str], set[str]]:
    primary: set[str] = set()
    extended: set[str] = set()
    for item in module_items:
        module = item["FILE_NAME"].replace("\\", "/")
        primary.update(f"{module}::{line}" for line in item["LINES"])
        extended.update(f"{module}::{line}" for line in item["EXTENDED_LINES"])
    return primary, extended - primary


def indices_for(entities: list[str], targets: set[str]) -> np.ndarray:
    return np.asarray(
        [index for index, entity in enumerate(entities) if entity in targets], dtype=int
    )


def score_policy(scores: np.ndarray, indices: np.ndarray) -> dict | None:
    if not indices.size:
        return None
    ranks = average_ranks_desc(scores)
    inspected = tie_inspection_sizes(scores)
    population = scores.size
    best_rank = float(np.min(ranks[indices]))
    best_inspected = int(np.min(inspected[indices]))
    target_count = int(indices.size)
    single_gmar = (population - inspected[indices]) / population
    return {
        "rank": best_rank,
        "mrr": 1.0 / best_rank,
        "exam": best_rank / population,
        "gmar_f": float(gmar_score(best_inspected, target_count, population)),
        "gmar_a": float(np.mean(single_gmar)),
        **{f"top{k}": float(best_rank <= k) for k in TOP_KS},
    }


def summarize_policy(rows: list[dict], policy: str) -> dict:
    available = [row[policy] for row in rows if row[policy] is not None]
    result = {"evaluable": len(available)}
    for metric in METRICS:
        values = [item[metric] for item in available]
        result[metric] = float(statistics.mean(values)) if values else None
    return result


def summarize_contrast(rows: list[dict], seed: int) -> dict:
    pairs = [row for row in rows if row["primary"] and row["expanded"]]
    result = {
        "paired": len(pairs),
        "primary_only": sum(bool(row["primary"]) and not row["expanded"] for row in rows),
        "expanded_only": sum(bool(row["expanded"]) and not row["primary"] for row in rows),
        "neither": sum(not row["primary"] and not row["expanded"] for row in rows),
        "metrics": {},
        "topk": {},
    }
    for offset, metric in enumerate(("mrr", "exam", "gmar_f", "gmar_a")):
        if metric == "exam":
            deltas = [row["primary"][metric] - row["expanded"][metric] for row in pairs]
            definition = "primary_minus_expanded; positive favors expanded"
        else:
            deltas = [row["expanded"][metric] - row["primary"][metric] for row in pairs]
            definition = "expanded_minus_primary; positive favors expanded"
        result["metrics"][metric] = {
            "mean_effect": float(statistics.mean(deltas)) if deltas else None,
            "bootstrap_95ci": percentile_interval(deltas, seed + offset),
            "effect_definition": definition,
        }
    for k in TOP_KS:
        gains = sum(row["primary"][f"top{k}"] < row["expanded"][f"top{k}"] for row in pairs)
        losses = sum(row["primary"][f"top{k}"] > row["expanded"][f"top{k}"] for row in pairs)
        result["topk"][str(k)] = {
            "gains": gains,
            "losses": losses,
            "net": gains - losses,
            "two_sided_exact_sign_p": exact_sign_p(gains, losses),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("subject_csv", type=Path)
    parser.add_argument("ground_truth_json", type=Path)
    parser.add_argument("rankings", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--replicates", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replicates < 100:
        raise ValueError("At least 100 randomization replicates are required")

    with args.subject_csv.open(encoding="utf-8-sig", newline="") as handle:
        subjects = list(csv.DictReader(handle))
    ground_truth = json.loads(args.ground_truth_json.read_text(encoding="utf-8"))
    rows_by_formula: dict[str, list[dict]] = {formula: [] for formula in FORMULAS}
    placebo_by_formula = {
        formula: empty_accumulator(args.replicates) for formula in FORMULAS
    }
    project_placebo: dict[str, dict[str, dict]] = defaultdict(dict)
    entity_set_mismatches = []
    missing_files = []
    eligible_ids: dict[str, list[str]] = {formula: [] for formula in FORMULAS}

    for subject in subjects:
        project = subject["BENCHMARK_NAME"]
        bug = subject["BUG_NUMBER"]
        bug_key = f"{project}:{bug}"
        if bug_key not in ground_truth:
            raise KeyError(f"Missing ground truth: {bug_key}")
        primary_targets, extra_targets = target_sets(ground_truth[bug_key])
        reference_entities = None
        for formula in FORMULAS:
            path = ranking_path(args.rankings, project, bug, formula)
            if not path.is_file():
                missing_files.append(str(path))
                continue
            entities, scores = load_ranking(path)
            if reference_entities is None:
                reference_entities = set(entities)
            elif set(entities) != reference_entities:
                entity_set_mismatches.append({"bug_id": bug_key, "formula": formula})
            primary_indices = indices_for(entities, primary_targets)
            extra_indices = indices_for(entities, extra_targets)
            expanded_indices = np.asarray(
                sorted(set(map(int, primary_indices)) | set(map(int, extra_indices))),
                dtype=int,
            )
            row = {
                "bug_id": bug_key,
                "project": project,
                "published_primary_targets": len(primary_targets),
                "published_extra_targets": len(extra_targets),
                "mapped_primary_targets": int(primary_indices.size),
                "mapped_extra_targets": int(extra_indices.size),
                "ranking_size": len(entities),
                "primary": score_policy(scores, primary_indices),
                "expanded": score_policy(scores, expanded_indices),
            }
            rows_by_formula[formula].append(row)

            extra_only = np.asarray(
                sorted(set(map(int, extra_indices)) - set(map(int, primary_indices))),
                dtype=int,
            )
            if not extra_only.size:
                continue
            all_official = set(map(int, expanded_indices))
            module_counts: Counter[str] = Counter(
                entities[index].rsplit("::", 1)[0] for index in extra_only
            )
            pools = {}
            if any(
                (
                    pool := np.asarray(
                        [
                            index
                            for index, entity in enumerate(entities)
                            if entity.rsplit("::", 1)[0] == module
                            and index not in all_official
                        ],
                        dtype=int,
                    )
                ).size
                < count
                or not pools.setdefault(module, pool).size
                for module, count in module_counts.items()
            ):
                continue
            rng = np.random.default_rng(stable_seed(args.seed, f"{bug_key}:{formula}"))
            sampled = sample_class_matched(
                rng, pools, dict(module_counts), args.replicates
            )
            observed, placebo = metric_values(
                scores, expanded_indices, primary_indices, sampled
            )
            add_result(
                placebo_by_formula[formula],
                observed,
                placebo,
                int(expanded_indices.size),
            )
            if formula not in project_placebo[project]:
                project_placebo[project][formula] = empty_accumulator(args.replicates)
            add_result(
                project_placebo[project][formula],
                observed,
                placebo,
                int(expanded_indices.size),
            )
            eligible_ids[formula].append(bug_key)

    if missing_files:
        raise FileNotFoundError(f"Missing {len(missing_files)} ranking files")
    if entity_set_mismatches:
        raise RuntimeError(f"Formula ranking universes differ: {entity_set_mismatches[:3]}")

    policy_summary = {
        formula: {
            "primary": summarize_policy(rows, "primary"),
            "expanded": summarize_policy(rows, "expanded"),
            "contrast": summarize_contrast(rows, args.seed + index * 100),
        }
        for index, (formula, rows) in enumerate(rows_by_formula.items())
    }
    placebo_summary = {
        "all": {
            formula: summarize_accumulator(accumulator, args.replicates)
            for formula, accumulator in placebo_by_formula.items()
        },
        "by_project": {
            project: {
                formula: summarize_accumulator(accumulator, args.replicates)
                for formula, accumulator in formula_results.items()
            }
            for project, formula_results in sorted(project_placebo.items())
        },
    }
    placebo_tests = [
        (
            f"{formula}:{metric}",
            result["metrics"][metric]["one_sided_randomization_p"],
        )
        for formula, result in placebo_summary["all"].items()
        for metric in METRICS
    ]
    placebo_holm = holm_adjust(placebo_tests)
    for formula, result in placebo_summary["all"].items():
        for metric in METRICS:
            result["metrics"][metric]["holm_adjusted_p_across_24"] = placebo_holm[
                f"{formula}:{metric}"
            ]
    policy_top_tests = [
        (
            f"{formula}:top{k}",
            result["contrast"]["topk"][str(k)]["two_sided_exact_sign_p"],
        )
        for formula, result in policy_summary.items()
        for k in TOP_KS
    ]
    policy_holm = holm_adjust(policy_top_tests)
    for formula, result in policy_summary.items():
        for k in TOP_KS:
            result["contrast"]["topk"][str(k)][
                "holm_adjusted_p_across_12"
            ] = policy_holm[f"{formula}:top{k}"]
    selected_keys = {
        f"{row['BENCHMARK_NAME']}:{row['BUG_NUMBER']}" for row in subjects
    }
    extra_candidate_bugs = sum(
        any(item["EXTENDED_LINES"] for item in ground_truth[key])
        for key in selected_keys
    )
    sanity = {
        "subject_count_is_135": len(subjects) == 135,
        "all_405_selected_rankings_present": sum(len(rows) for rows in rows_by_formula.values())
        == 405,
        "formula_entity_sets_identical_per_bug": not entity_set_mismatches,
        "expanded_never_less_evaluable": all(
            result["expanded"]["evaluable"] >= result["primary"]["evaluable"]
            for result in policy_summary.values()
        ),
        "expanded_first_hit_never_worsens": all(
            result["contrast"]["topk"][str(k)]["losses"] == 0
            for result in policy_summary.values()
            for k in TOP_KS
        ),
        "same_placebo_bug_ids_across_formulas": len(
            {tuple(ids) for ids in eligible_ids.values()}
        )
        == 1,
    }
    sanity["passed"] = all(sanity.values())
    if not sanity["passed"]:
        raise RuntimeError(f"External validation checks failed: {sanity}")

    ranking_hash = hashlib.sha256()
    for path in sorted(args.rankings.glob("*.csv")):
        if any(f"_{formula}_Statement.csv" in path.name for formula in FORMULAS):
            ranking_hash.update(path.name.encode())
            ranking_hash.update(path.read_bytes())
    output = {
        "provenance": {
            "dataset": "BugsInPy real Python bugs via FauxPy EMSE 2024 replication package",
            "source_commit": args.source_commit,
            "license": "GPL-3.0",
            "subject_csv_sha256": file_sha256(args.subject_csv),
            "ground_truth_json_sha256": file_sha256(args.ground_truth_json),
            "three_formula_rankings_tree_sha256": ranking_hash.hexdigest(),
        },
        "design": {
            "primary_policy": "published LINES: removed/modified buggy lines plus the scope-aware executable location following an insertion",
            "expanded_policy": "published LINES plus EXTENDED_LINES: add the scope-aware executable location preceding an insertion",
            "ranking_universe": "entities present in each published FauxPy statement ranking CSV; ties use average rank for conventional metrics and tie-group-end inspection for GMAR",
            "negative_control": "keep mapped LINES targets fixed and replace mapped EXTENDED_LINES targets without replacement by non-target ranked entities from the same Python module; preserve per-module counts",
            "replicates": args.replicates,
            "seed": args.seed,
            "limitation": "EXTENDED_LINES identifies a conservative subset of insertion-related alternative proxies; the released JSON does not retain a complete per-bug omission flag",
        },
        "scope": {
            "subjects": len(subjects),
            "projects": len({row["BENCHMARK_NAME"] for row in subjects}),
            "formulas": list(FORMULAS),
            "ranking_files": sum(len(rows) for rows in rows_by_formula.values()),
            "bugs_with_published_extra_targets": extra_candidate_bugs,
            "placebo_eligible_bugs": {
                formula: len(ids) for formula, ids in eligible_ids.items()
            },
        },
        "sanity_checks": sanity,
        "policy_summary": policy_summary,
        "placebo_summary": placebo_summary,
        "placebo_eligible_bug_ids": eligible_ids,
        "rows": rows_by_formula,
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
                "policy_summary": policy_summary,
                "placebo_all": placebo_summary["all"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
