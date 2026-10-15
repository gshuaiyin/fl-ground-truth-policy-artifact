"""Decompose deleted-only, official-direct, and omission-aware policy effects."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from analyze_groundtruth_sensitivity import holm_adjust
from project_robustness_analysis import exact_sign_p, percentile_interval


FORMULAS = ("ochiai", "dstar", "tarantula", "barinel")
TOP_KS = (1, 5, 10, 20)
POLICIES = ("deleted_only", "official_direct", "official_omission_aware")
CONTRASTS = (
    ("deleted_to_direct", "deleted_only", "official_direct"),
    ("direct_to_omission", "official_direct", "official_omission_aware"),
    ("deleted_to_omission", "deleted_only", "official_omission_aware"),
)


def rank_key(policy: str, formula: str) -> str:
    return f"line_{policy}_{formula}_rank"


def summarize_policy(rows: list[dict], policy: str, formula: str) -> dict:
    ranks = [
        float(row[rank_key(policy, formula)])
        for row in rows
        if row[rank_key(policy, formula)] is not None
    ]
    return {
        "evaluable": len(ranks),
        "median_rank": statistics.median(ranks) if ranks else None,
        "mrr": statistics.mean(1.0 / rank for rank in ranks) if ranks else None,
        **{f"top{k}": sum(rank <= k for rank in ranks) for k in TOP_KS},
    }


def contrast(rows: list[dict], left: str, right: str, formula: str, seed: int) -> dict:
    left_key = rank_key(left, formula)
    right_key = rank_key(right, formula)
    paired = [
        row
        for row in rows
        if row[left_key] is not None and row[right_key] is not None
    ]
    omission = [row for row in paired if row["has_omission_candidates"]]

    def deltas(group: list[dict], metric: str) -> list[float]:
        if metric == "mrr":
            return [
                1.0 / float(row[right_key]) - 1.0 / float(row[left_key])
                for row in group
            ]
        return [
            float(row[left_key]) / int(row["line_ranking_size"])
            - float(row[right_key]) / int(row["line_ranking_size"])
            for row in group
        ]

    output = {
        "evaluable": {
            "left": sum(row[left_key] is not None for row in rows),
            "left_only_bug_ids": sorted(
                row["bug_id"]
                for row in rows
                if row[left_key] is not None and row[right_key] is None
            ),
            "right": sum(row[right_key] is not None for row in rows),
            "right_only_bug_ids": sorted(
                row["bug_id"]
                for row in rows
                if row[right_key] is not None and row[left_key] is None
            ),
        },
        "paired": len(paired),
        "paired_omission": len(omission),
    }
    output["evaluable"]["net_gain"] = (
        output["evaluable"]["right"] - output["evaluable"]["left"]
    )
    for offset, metric in enumerate(("mrr", "exam_improvement")):
        all_values = deltas(paired, metric)
        omission_values = deltas(omission, metric)
        output[metric] = {
            "all_paired_bootstrap_95ci": percentile_interval(
                all_values, seed + offset
            ),
            "all_paired_mean": statistics.mean(all_values) if all_values else None,
            "omission_paired_bootstrap_95ci": percentile_interval(
                omission_values, seed + 10 + offset
            ),
            "omission_paired_mean": statistics.mean(omission_values)
            if omission_values
            else None,
        }
    output["topk"] = {}
    for k in TOP_KS:
        gains = sum(
            float(row[left_key]) > k >= float(row[right_key]) for row in paired
        )
        losses = sum(
            float(row[right_key]) > k >= float(row[left_key]) for row in paired
        )
        output["topk"][str(k)] = {
            "exact_sign_p": exact_sign_p(gains, losses),
            "gains": gains,
            "losses": losses,
            "net": gains - losses,
        }
    output["rank_improvements"] = sum(
        float(row[right_key]) < float(row[left_key]) for row in paired
    )
    output["rank_ties"] = sum(
        float(row[right_key]) == float(row[left_key]) for row in paired
    )
    output["rank_worsenings"] = sum(
        float(row[right_key]) > float(row[left_key]) for row in paired
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = data["rows"]
    output: dict[str, object] = {
        "analysis_entity": "source_line",
        "bootstrap_replicates": 20_000,
        "contrasts": {},
        "n_bugs": len(rows),
        "policy_summary": {},
        "sanity_checks": {},
    }
    for policy in POLICIES:
        output["policy_summary"][policy] = {
            formula: summarize_policy(rows, policy, formula) for formula in FORMULAS
        }

    all_topk_tests = []
    for contrast_index, (name, left, right) in enumerate(CONTRASTS):
        output["contrasts"][name] = {
            "left_policy": left,
            "right_policy": right,
            "by_formula": {},
        }
        within_tests = []
        for formula_index, formula in enumerate(FORMULAS):
            result = contrast(
                rows,
                left,
                right,
                formula,
                20263000 + contrast_index * 100 + formula_index * 20,
            )
            output["contrasts"][name]["by_formula"][formula] = result
            for k in TOP_KS:
                comparison = result["topk"][str(k)]
                test = {
                    "contrast": name,
                    "formula": formula,
                    "k": k,
                    "raw_p": comparison["exact_sign_p"],
                }
                within_tests.append(test)
                all_topk_tests.append(test)
        within_adjusted = holm_adjust([float(test["raw_p"]) for test in within_tests])
        for test, adjusted in zip(within_tests, within_adjusted):
            result = output["contrasts"][name]["by_formula"][test["formula"]]
            result["topk"][str(test["k"])]["holm_p_within_16"] = adjusted

    global_adjusted = holm_adjust([float(test["raw_p"]) for test in all_topk_tests])
    for test, adjusted in zip(all_topk_tests, global_adjusted):
        result = output["contrasts"][test["contrast"]]["by_formula"][test["formula"]]
        result["topk"][str(test["k"])]["holm_p_global_48"] = adjusted

    direct_to_omission = output["contrasts"]["direct_to_omission"]["by_formula"]
    non_omission_differences = {}
    for formula in FORMULAS:
        left_key = rank_key("official_direct", formula)
        right_key = rank_key("official_omission_aware", formula)
        non_omission_differences[formula] = sum(
            row[left_key] != row[right_key]
            for row in rows
            if not row["has_omission_candidates"]
        )
    output["sanity_checks"] = {
        "direct_to_omission_rank_worsenings": {
            formula: direct_to_omission[formula]["rank_worsenings"]
            for formula in FORMULAS
        },
        "non_omission_rank_differences_direct_vs_omission": non_omission_differences,
        "passed": all(
            direct_to_omission[formula]["rank_worsenings"] == 0
            and non_omission_differences[formula] == 0
            for formula in FORMULAS
        ),
    }
    if not output["sanity_checks"]["passed"]:
        raise RuntimeError(
            "Policy decomposition invariants failed; refusing to write results"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    compact = {
        "sanity_checks": output["sanity_checks"],
        "contrasts": {
            name: {
                formula: {
                    "evaluable": {
                        "left": result["evaluable"]["left"],
                        "right": result["evaluable"]["right"],
                        "net_gain": result["evaluable"]["net_gain"],
                        "left_only": len(
                            result["evaluable"]["left_only_bug_ids"]
                        ),
                        "right_only": len(
                            result["evaluable"]["right_only_bug_ids"]
                        ),
                    },
                    "mrr": result["mrr"],
                    "rank_improvements": result["rank_improvements"],
                    "rank_worsenings": result["rank_worsenings"],
                    "top10": result["topk"]["10"],
                }
                for formula, result in details["by_formula"].items()
            }
            for name, details in output["contrasts"].items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
