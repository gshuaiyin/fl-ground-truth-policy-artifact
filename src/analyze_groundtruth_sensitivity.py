"""Paired inference for the ground-truth sensitivity audit."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from gmar import random_hit_probability

FORMULAS = ("ochiai", "dstar", "tarantula", "barinel")
TOP_KS = (1, 5, 10, 20)


def interval(values: list[float], seed: int, replicates: int = 20_000) -> list[float]:
    rng = random.Random(seed)
    samples = sorted(
        statistics.mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(replicates)
    )
    return [samples[499], samples[19_499]]


def difference_interval(
    left: list[float], right: list[float], seed: int, replicates: int = 20_000
) -> list[float]:
    rng = random.Random(seed)
    samples = sorted(
        statistics.mean(left[rng.randrange(len(left))] for _ in left)
        - statistics.mean(right[rng.randrange(len(right))] for _ in right)
        for _ in range(replicates)
    )
    return [samples[499], samples[19_499]]


def project_stratified_difference(
    rows: list[dict], seed: int, replicates: int = 20_000
) -> tuple[float, list[float], dict[str, dict[str, float | int]]]:
    """Equal-project high-minus-low contrast with within-stratum bootstrap."""
    strata: dict[str, tuple[list[float], list[float]]] = {}
    details: dict[str, dict[str, float | int]] = {}
    for project in sorted({str(row["project"]) for row in rows}):
        high = [
            float(row["gmar"])
            for row in rows
            if row["project"] == project and int(row["mapped_targets"]) >= 4
        ]
        low = [
            float(row["gmar"])
            for row in rows
            if row["project"] == project and int(row["mapped_targets"]) <= 3
        ]
        if not high or not low:
            continue
        strata[project] = (high, low)
        details[project] = {
            "difference": statistics.mean(high) - statistics.mean(low),
            "n_high": len(high),
            "n_low": len(low),
        }
    observed = statistics.mean(
        statistics.mean(high) - statistics.mean(low) for high, low in strata.values()
    )
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        samples.append(
            statistics.mean(
                statistics.mean(high[rng.randrange(len(high))] for _ in high)
                - statistics.mean(low[rng.randrange(len(low))] for _ in low)
                for high, low in strata.values()
            )
        )
    samples.sort()
    return observed, [samples[499], samples[19_499]], details


def exact_sign_p(left: int, right: int) -> float:
    discordant = left + right
    if discordant == 0:
        return 1.0
    return min(
        1.0,
        2
        * sum(math.comb(discordant, i) for i in range(min(left, right) + 1))
        / 2**discordant,
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for order, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        running = max(running, min(1.0, (len(p_values) - order) * p_values[index]))
        adjusted[index] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    prefix = "line_" if "line_ranking_size" in data["rows"][0] else ""
    entity = "source_line" if prefix else "probe"
    deleted_key = f"{prefix}deleted_only_ochiai_rank"
    official_key = f"{prefix}official_omission_aware_ochiai_rank"
    inspected_key = f"{prefix}official_omission_aware_ochiai_tie_expanded_size"
    mapped_key = "mapped_official_lines" if prefix else "mapped_official_probes"
    population_key = f"{prefix}ranking_size"
    paired = [
        row
        for row in data["rows"]
        if row[deleted_key] is not None and row[official_key] is not None
    ]
    gains = sum(
        row[deleted_key] > 10 and row[official_key] <= 10
        for row in paired
    )
    losses = sum(
        row[deleted_key] <= 10 and row[official_key] > 10
        for row in paired
    )
    sign_p = exact_sign_p(gains, losses)
    omission = [row for row in paired if row["has_omission_candidates"]]
    deltas = [
        1 / row[official_key] - 1 / row[deleted_key]
        for row in omission
    ]
    gmar_rows = [
        row
        for row in data["rows"]
        if row[official_key] is not None and int(row[mapped_key]) > 0
    ]
    for row in gmar_rows:
        probability = random_hit_probability(
            int(row[inspected_key]), int(row[mapped_key]), int(row[population_key])
        )
        row["gmar"] = 1.0 - probability
        row["mapped_targets"] = int(row[mapped_key])
    gmar_strata = {}
    for flag in (False, True):
        group = [row for row in gmar_rows if bool(row["has_omission_candidates"]) == flag]
        gmar_strata[str(flag).lower()] = {
            "mean_gmar": statistics.mean(float(row["gmar"]) for row in group),
            "mean_mapped_targets": statistics.mean(
                int(row[mapped_key]) for row in group
            ),
            "n": len(group),
            "top10_rate": statistics.mean(
                float(row[official_key]) <= 10 for row in group
            ),
        }
    omission_gmar = [
        float(row["gmar"]) for row in gmar_rows if row["has_omission_candidates"]
    ]
    direct_gmar = [
        float(row["gmar"]) for row in gmar_rows if not row["has_omission_candidates"]
    ]
    gmar_bins = {}
    for label, predicate in (
        ("one", lambda count: count == 1),
        ("two_or_three", lambda count: 2 <= count <= 3),
        ("four_or_more", lambda count: count >= 4),
    ):
        group = [
            row
            for row in gmar_rows
            if row["has_omission_candidates"]
            and predicate(int(row[mapped_key]))
        ]
        observed_top10 = statistics.mean(float(row[official_key]) <= 10 for row in group)
        random_top10 = statistics.mean(
            random_hit_probability(
                min(10, int(row[population_key])),
                int(row[mapped_key]),
                int(row[population_key]),
            )
            for row in group
        )
        gmar_bins[label] = {
            "mean_gmar": statistics.mean(float(row["gmar"]) for row in group),
            "mean_observed_exam": statistics.mean(
                float(row[official_key]) / int(row[population_key]) for row in group
            ),
            "mean_random_expected_exam": statistics.mean(
                (int(row[population_key]) + 1)
                / (int(row[mapped_key]) + 1)
                / int(row[population_key])
                for row in group
            ),
            "mean_random_top10_rate": random_top10,
            "n": len(group),
            "observed_minus_random_top10_rate": observed_top10 - random_top10,
            "top10_rate": observed_top10,
        }
    low_multiplicity = [
        float(row["gmar"])
        for row in gmar_rows
        if row["has_omission_candidates"]
        and int(row[mapped_key]) <= 3
    ]
    high_multiplicity = [
        float(row["gmar"])
        for row in gmar_rows
        if row["has_omission_candidates"]
        and int(row[mapped_key]) >= 4
    ]
    omission_rows = [row for row in gmar_rows if row["has_omission_candidates"]]
    project_difference, project_ci, project_details = project_stratified_difference(
        omission_rows, 20260827
    )
    formula_results = []
    for formula in FORMULAS:
        rank_key = f"{prefix}official_omission_aware_{formula}_rank"
        formula_inspected_key = (
            f"{prefix}official_omission_aware_{formula}_tie_expanded_size"
        )
        group = [
            row
            for row in data["rows"]
            if row[rank_key] is not None and int(row[mapped_key]) > 0
        ]
        formula_results.append(
            {
                "formula": formula,
                "mean_gmar_average": statistics.mean(
                    float(
                        row[
                            f"line_official_omission_aware_{formula}_gmar_average"
                        ]
                    )
                    for row in group
                ) if prefix else None,
                "mean_gmar": statistics.mean(
                    1.0
                    - random_hit_probability(
                        int(row[formula_inspected_key]),
                        int(row[mapped_key]),
                        int(row[population_key]),
                    )
                    for row in group
                ),
                "mrr": statistics.mean(1.0 / float(row[rank_key]) for row in group),
                "n": len(group),
                "top10_rate": statistics.mean(float(row[rank_key]) <= 10 for row in group),
            }
        )
    conventional_order = [
        row["formula"]
        for row in sorted(formula_results, key=lambda row: row["mrr"], reverse=True)
    ]
    gmar_order = [
        row["formula"]
        for row in sorted(formula_results, key=lambda row: row["mean_gmar"], reverse=True)
    ]
    gmar_average_order = (
        [
            row["formula"]
            for row in sorted(
                formula_results,
                key=lambda row: float(row["mean_gmar_average"]),
                reverse=True,
            )
        ]
        if prefix
        else []
    )
    gmar_average_result = None
    if prefix:
        average_key = "line_official_omission_aware_ochiai_gmar_average"
        average_rows = [
            {**row, "gmar": float(row[average_key])}
            for row in gmar_rows
            if row[average_key] is not None
        ]
        average_high = [
            float(row["gmar"])
            for row in average_rows
            if row["has_omission_candidates"] and int(row[mapped_key]) >= 4
        ]
        average_low = [
            float(row["gmar"])
            for row in average_rows
            if row["has_omission_candidates"] and int(row[mapped_key]) <= 3
        ]
        average_omission_rows = [
            row for row in average_rows if row["has_omission_candidates"]
        ]
        avg_project_diff, avg_project_ci, avg_project_details = (
            project_stratified_difference(average_omission_rows, 20260828)
        )
        average_formula_deltas = [
            float(row["line_official_omission_aware_ochiai_gmar_average"])
            - float(row["line_official_omission_aware_dstar_gmar_average"])
            for row in data["rows"]
            if row["line_official_omission_aware_ochiai_gmar_average"] is not None
            and row["line_official_omission_aware_dstar_gmar_average"] is not None
        ]
        gmar_average_result = {
            "formula_order": gmar_average_order,
            "paired_ochiai_minus_dstar": {
                "bootstrap_95ci": interval(average_formula_deltas, 20260830),
                "mean_difference": statistics.mean(average_formula_deltas),
                "n": len(average_formula_deltas),
            },
            "high_minus_low_multiplicity": {
                "bootstrap_95ci": difference_interval(
                    average_high, average_low, 20260829
                ),
                "equal_project_weighted_bootstrap_95ci": avg_project_ci,
                "equal_project_weighted_mean_difference": avg_project_diff,
                "mean_difference": statistics.mean(average_high)
                - statistics.mean(average_low),
                "projects": avg_project_details,
            },
        }
    policy_summary = {}
    for policy in ("deleted_only", "official_direct", "official_omission_aware"):
        policy_summary[policy] = {}
        for formula in FORMULAS:
            rank_key = f"{prefix}{policy}_{formula}_rank"
            ranks = [
                float(row[rank_key])
                for row in data["rows"]
                if row[rank_key] is not None
            ]
            policy_summary[policy][formula] = {
                "evaluable": len(ranks),
                "median_rank": statistics.median(ranks),
                "mrr": statistics.mean(1.0 / rank for rank in ranks),
                "top1": sum(rank <= 1 for rank in ranks),
                "top5": sum(rank <= 5 for rank in ranks),
                "top10": sum(rank <= 10 for rank in ranks),
                "top20": sum(rank <= 20 for rank in ranks),
            }
    topk_comparisons = []
    formula_mrr = {}
    for formula in FORMULAS:
        deleted_formula_key = f"{prefix}deleted_only_{formula}_rank"
        official_formula_key = f"{prefix}official_omission_aware_{formula}_rank"
        formula_paired = [
            row
            for row in data["rows"]
            if row[deleted_formula_key] is not None
            and row[official_formula_key] is not None
        ]
        formula_omission = [
            row for row in formula_paired if row["has_omission_candidates"]
        ]
        formula_deltas = [
            1.0 / float(row[official_formula_key])
            - 1.0 / float(row[deleted_formula_key])
            for row in formula_omission
        ]
        formula_mrr[formula] = {
            "bootstrap_95ci": interval(formula_deltas, 20260900 + len(formula_mrr)),
            "mean_delta": statistics.mean(formula_deltas),
            "n": len(formula_deltas),
        }
        for k in TOP_KS:
            formula_gains = sum(
                float(row[deleted_formula_key]) > k
                and float(row[official_formula_key]) <= k
                for row in formula_paired
            )
            formula_losses = sum(
                float(row[deleted_formula_key]) <= k
                and float(row[official_formula_key]) > k
                for row in formula_paired
            )
            topk_comparisons.append(
                {
                    "formula": formula,
                    "gains": formula_gains,
                    "k": k,
                    "losses": formula_losses,
                    "n": len(formula_paired),
                    "net": formula_gains - formula_losses,
                    "raw_p": exact_sign_p(formula_gains, formula_losses),
                }
            )
    adjusted = holm_adjust([float(row["raw_p"]) for row in topk_comparisons])
    for row, adjusted_p in zip(topk_comparisons, adjusted):
        row["holm_p_16_tests"] = adjusted_p
    result = {
        "analysis_entity": entity,
        "gmar": {
            "definition": "1 - P(a random ranking hits at least one of m mapped targets within the observed tie-expanded inspection set)",
            "formula_results": formula_results,
            "formula_order_by_gmar": gmar_order,
            "formula_order_by_mrr": conventional_order,
            "formula_order_changed": gmar_order != conventional_order,
            "high_minus_low_multiplicity": {
                "bootstrap_95ci": difference_interval(
                    high_multiplicity, low_multiplicity, 20260826
                ),
                "mean_difference": statistics.mean(high_multiplicity)
                - statistics.mean(low_multiplicity),
                "n_high": len(high_multiplicity),
                "n_low": len(low_multiplicity),
                "equal_project_weighted_bootstrap_95ci": project_ci,
                "equal_project_weighted_mean_difference": project_difference,
                "projects": project_details,
            },
            "omission_minus_non_omission_mean_difference": statistics.mean(
                omission_gmar
            )
            - statistics.mean(direct_gmar),
            "omission_minus_non_omission_bootstrap_95ci": difference_interval(
                omission_gmar, direct_gmar, 20260825
            ),
            "omission_mapped_target_bins": gmar_bins,
            "strata": gmar_strata,
        },
        "gmar_average": gmar_average_result,
        "omission_paired": {
            "by_formula": formula_mrr,
            "mrr_delta": statistics.mean(deltas),
            "mrr_delta_bug_bootstrap_95ci": interval(deltas, 20260824),
            "n": len(omission),
        },
        "paired_top10": {
            "exact_two_sided_p": sign_p,
            "gains": gains,
            "losses": losses,
            "n": len(paired),
            "net": gains - losses,
        },
        "paired_topk_all_formulas": topk_comparisons,
        "policy_summary": policy_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
