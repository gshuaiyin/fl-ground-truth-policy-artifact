"""Sensitivity of GMAR conclusions to three suspiciousness-tie policies."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from analyze_groundtruth_sensitivity import difference_interval, interval
from gmar import gmar_score


FORMULAS = ("ochiai", "dstar", "tarantula", "barinel")


def inspection_sizes(rank: float, tie_end: int) -> dict[str, int]:
    tie_start = int(round(2.0 * rank - tie_end))
    if not 1 <= tie_start <= tie_end:
        raise ValueError(f"invalid tie interval: rank={rank}, end={tie_end}")
    return {
        "optimistic_tie_start": tie_start,
        "midrank_ceiling": math.ceil(rank),
        "full_tie_expanded": tie_end,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))

    formula_rows = []
    primary_by_policy: dict[str, list[dict]] = {}
    scores_by_formula_policy: dict[tuple[str, str], dict[str, float]] = {}
    for formula in FORMULAS:
        rank_key = f"line_official_omission_aware_{formula}_rank"
        end_key = f"line_official_omission_aware_{formula}_tie_expanded_size"
        rows = [
            row
            for row in data["rows"]
            if row[rank_key] is not None and int(row["mapped_official_lines"]) > 0
        ]
        scores: dict[str, list[float]] = {
            name: [] for name in inspection_sizes(1.0, 1)
        }
        scored_rows: dict[str, list[dict]] = {name: [] for name in scores}
        tied = 0
        for row in rows:
            sizes = inspection_sizes(float(row[rank_key]), int(row[end_key]))
            tied += sizes["optimistic_tie_start"] != sizes["full_tie_expanded"]
            for policy, inspected in sizes.items():
                score = gmar_score(
                    inspected,
                    int(row["mapped_official_lines"]),
                    int(row["line_ranking_size"]),
                )
                scores[policy].append(score)
                scored_rows[policy].append({**row, "gmar": score})
        for policy, values in scores.items():
            scores_by_formula_policy[(formula, policy)] = {
                str(row["bug_id"]): float(scored["gmar"])
                for row, scored in zip(rows, scored_rows[policy])
            }
            formula_rows.append(
                {
                    "formula": formula,
                    "mean_gmar": statistics.mean(values),
                    "n": len(values),
                    "policy": policy,
                    "tied_first_target_rate": tied / len(rows),
                }
            )
        if formula == "ochiai":
            primary_by_policy = scored_rows

    orders = {
        policy: [
            row["formula"]
            for row in sorted(
                (row for row in formula_rows if row["policy"] == policy),
                key=lambda row: float(row["mean_gmar"]),
                reverse=True,
            )
        ]
        for policy in primary_by_policy
    }
    multiplicity = {}
    for index, (policy, rows) in enumerate(primary_by_policy.items()):
        high = [
            float(row["gmar"])
            for row in rows
            if row["has_omission_candidates"]
            and int(row["mapped_official_lines"]) >= 4
        ]
        low = [
            float(row["gmar"])
            for row in rows
            if row["has_omission_candidates"]
            and int(row["mapped_official_lines"]) <= 3
        ]
        multiplicity[policy] = {
            "bootstrap_95ci": difference_interval(high, low, 20261200 + index),
            "high_minus_low_mean": statistics.mean(high) - statistics.mean(low),
            "n_high": len(high),
            "n_low": len(low),
        }
    primary_formula_contrast = {}
    for index, policy in enumerate(primary_by_policy):
        dstar = scores_by_formula_policy[("dstar", policy)]
        ochiai = scores_by_formula_policy[("ochiai", policy)]
        bug_ids = sorted(dstar.keys() & ochiai.keys())
        deltas = [dstar[bug_id] - ochiai[bug_id] for bug_id in bug_ids]
        primary_formula_contrast[policy] = {
            "bootstrap_95ci": interval(deltas, 20261300 + index),
            "interpretation": "positive means DStar is better than Ochiai",
            "mean_dstar_minus_ochiai": statistics.mean(deltas),
            "n": len(deltas),
        }
    output = {
        "formula_orders": orders,
        "formula_policy_results": formula_rows,
        "ochiai_high_minus_low_multiplicity": multiplicity,
        "paired_dstar_minus_ochiai": primary_formula_contrast,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
