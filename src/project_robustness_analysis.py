"""Project-level and leave-one-project-out robustness for ground-truth policies."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from gmar import gmar_score


FORMULAS = ("ochiai", "dstar", "tarantula", "barinel")
TOP_KS = (1, 5, 10, 20)
BOOTSTRAP_REPLICATES = 20_000


def percentile_interval(
    values: list[float], seed: int, replicates: int = BOOTSTRAP_REPLICATES
) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    batch_size = 1_000
    for start in range(0, replicates, batch_size):
        stop = min(replicates, start + batch_size)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        samples[start:stop] = array[indices].mean(axis=1)
    samples.sort()
    return [
        float(samples[int(0.025 * replicates)]),
        float(samples[int(0.975 * replicates) - 1]),
    ]


def equal_project_interval(
    values: dict[str, list[float]], seed: int, replicates: int = BOOTSTRAP_REPLICATES
) -> tuple[float, list[float]]:
    nonempty = {project: group for project, group in values.items() if group}
    observed = statistics.mean(statistics.mean(group) for group in nonempty.values())
    rng = np.random.default_rng(seed)
    samples = np.zeros(replicates, dtype=float)
    batch_size = 1_000
    for group in nonempty.values():
        array = np.asarray(group, dtype=float)
        for start in range(0, replicates, batch_size):
            stop = min(replicates, start + batch_size)
            indices = rng.integers(0, array.size, size=(stop - start, array.size))
            samples[start:stop] += array[indices].mean(axis=1) / len(nonempty)
    samples.sort()
    return observed, [
        float(samples[int(0.025 * replicates)]),
        float(samples[int(0.975 * replicates) - 1]),
    ]


def exact_sign_p(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    return min(
        1.0,
        2.0
        * sum(math.comb(discordant, i) for i in range(min(gains, losses) + 1))
        / 2**discordant,
    )


def row_delta(row: dict, formula: str) -> dict[str, float]:
    deleted_rank = float(row[f"line_deleted_only_{formula}_rank"])
    official_rank = float(row[f"line_official_omission_aware_{formula}_rank"])
    population = int(row["line_ranking_size"])
    deleted_gmar = gmar_score(
        int(row[f"line_deleted_only_{formula}_tie_expanded_size"]),
        int(row["mapped_deleted_lines"]),
        population,
    )
    official_gmar = gmar_score(
        int(row[f"line_official_omission_aware_{formula}_tie_expanded_size"]),
        int(row["mapped_official_lines"]),
        population,
    )
    return {
        "exam_improvement": deleted_rank / population - official_rank / population,
        "gmar_f_delta": official_gmar - deleted_gmar,
        "mrr_delta": 1.0 / official_rank - 1.0 / deleted_rank,
    }


def paired_rows(rows: list[dict], formula: str) -> list[dict]:
    return [
        row
        for row in rows
        if row[f"line_deleted_only_{formula}_rank"] is not None
        and row[f"line_official_omission_aware_{formula}_rank"] is not None
    ]


def summarize_group(rows: list[dict], formula: str, seed: int) -> dict:
    pairs = paired_rows(rows, formula)
    omission = [row for row in pairs if row["has_omission_candidates"]]
    deltas = [row_delta(row, formula) for row in pairs]
    omission_deltas = [row_delta(row, formula) for row in omission]
    output = {
        "bugs": len(rows),
        "deleted_evaluable": sum(
            row[f"line_deleted_only_{formula}_rank"] is not None for row in rows
        ),
        "official_evaluable": sum(
            row[f"line_official_omission_aware_{formula}_rank"] is not None
            for row in rows
        ),
        "paired": len(pairs),
        "paired_omission": len(omission),
    }
    output["evaluability_gain"] = output["official_evaluable"] - output["deleted_evaluable"]
    for offset, metric in enumerate(("mrr_delta", "exam_improvement", "gmar_f_delta")):
        values = [delta[metric] for delta in deltas]
        omission_values = [delta[metric] for delta in omission_deltas]
        output[metric] = {
            "all_paired_bootstrap_95ci": percentile_interval(values, seed + offset),
            "all_paired_mean": statistics.mean(values) if values else None,
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
            float(row[f"line_deleted_only_{formula}_rank"]) > k
            >= float(row[f"line_official_omission_aware_{formula}_rank"])
            for row in pairs
        )
        losses = sum(
            float(row[f"line_official_omission_aware_{formula}_rank"]) > k
            >= float(row[f"line_deleted_only_{formula}_rank"])
            for row in pairs
        )
        output["topk"][str(k)] = {
            "exact_sign_p": exact_sign_p(gains, losses),
            "gains": gains,
            "losses": losses,
            "net": gains - losses,
        }
    return output


def dominance(project_results: dict[str, dict], k: int) -> dict:
    contributions = {
        project: {
            "gains": int(result["topk"][str(k)]["gains"]),
            "losses": int(result["topk"][str(k)]["losses"]),
            "net": int(result["topk"][str(k)]["net"]),
        }
        for project, result in project_results.items()
    }
    total_gains = sum(item["gains"] for item in contributions.values())
    total_losses = sum(item["losses"] for item in contributions.values())
    total_discordant = total_gains + total_losses
    largest_gain_project = max(
        contributions, key=lambda project: contributions[project]["gains"]
    )
    largest_discordant_project = max(
        contributions,
        key=lambda project: contributions[project]["gains"]
        + contributions[project]["losses"],
    )
    largest_positive_net_project = max(
        contributions, key=lambda project: contributions[project]["net"]
    )
    positive_net_total = sum(
        max(0, item["net"]) for item in contributions.values()
    )
    total_net = total_gains - total_losses
    return {
        "by_project": contributions,
        "largest_discordant_project": largest_discordant_project,
        "largest_discordant_share": (
            (
                contributions[largest_discordant_project]["gains"]
                + contributions[largest_discordant_project]["losses"]
            )
            / total_discordant
            if total_discordant
            else 0.0
        ),
        "largest_gain_project": largest_gain_project,
        "largest_gain_share": (
            contributions[largest_gain_project]["gains"] / total_gains
            if total_gains
            else 0.0
        ),
        "largest_positive_net_project": largest_positive_net_project,
        "largest_positive_net_share": (
            contributions[largest_positive_net_project]["net"]
            / positive_net_total
            if positive_net_total
            else 0.0
        ),
        "largest_net_over_total_net": (
            contributions[largest_positive_net_project]["net"] / total_net
            if total_net
            else None
        ),
        "positive_net_total": positive_net_total,
        "total_discordant": total_discordant,
        "total_gains": total_gains,
        "total_losses": total_losses,
        "total_net": total_net,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = data["rows"]
    projects = sorted({str(row["project"]) for row in rows})
    result: dict[str, object] = {
        "analysis_entity": "source_line",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "definitions": {
            "exam_improvement": "deleted-only EXAM minus omission-aware EXAM; positive favors omission-aware",
            "gmar_f_delta": "omission-aware GMAR-F minus deleted-only GMAR-F",
            "mrr_delta": "omission-aware reciprocal rank minus deleted-only reciprocal rank",
            "project_weighting": "bug-weighted results pool bugs; equal-project results average six project means",
        },
        "n_bugs": len(rows),
        "projects": projects,
        "by_formula": {},
    }
    for formula_index, formula in enumerate(FORMULAS):
        seed_base = 20262000 + formula_index * 1000
        per_project = {
            project: summarize_group(
                [row for row in rows if row["project"] == project],
                formula,
                seed_base + project_index * 50,
            )
            for project_index, project in enumerate(projects)
        }
        leave_one_out = {
            project: summarize_group(
                [row for row in rows if row["project"] != project],
                formula,
                seed_base + 400 + project_index * 50,
            )
            for project_index, project in enumerate(projects)
        }
        pairs = paired_rows(rows, formula)
        equal_project = {}
        for metric_index, metric in enumerate(
            ("mrr_delta", "exam_improvement", "gmar_f_delta")
        ):
            grouped_all = {
                project: [
                    row_delta(row, formula)[metric]
                    for row in pairs
                    if row["project"] == project
                ]
                for project in projects
            }
            grouped_omission = {
                project: [
                    row_delta(row, formula)[metric]
                    for row in pairs
                    if row["project"] == project
                    and row["has_omission_candidates"]
                ]
                for project in projects
            }
            all_mean, all_ci = equal_project_interval(
                grouped_all, seed_base + 800 + metric_index
            )
            omission_mean, omission_ci = equal_project_interval(
                grouped_omission, seed_base + 810 + metric_index
            )
            equal_project[metric] = {
                "all_paired_bootstrap_95ci": all_ci,
                "all_paired_mean": all_mean,
                "all_paired_project_means": {
                    project: statistics.mean(values)
                    for project, values in grouped_all.items()
                    if values
                },
                "omission_paired_bootstrap_95ci": omission_ci,
                "omission_paired_mean": omission_mean,
                "omission_paired_project_means": {
                    project: statistics.mean(values)
                    for project, values in grouped_omission.items()
                    if values
                },
            }
        formula_result = {
            "all_projects_pooled": summarize_group(rows, formula, seed_base + 900),
            "by_project": per_project,
            "equal_project_weighted": equal_project,
            "leave_one_project_out": leave_one_out,
            "topk_dominance": {
                str(k): dominance(per_project, k) for k in TOP_KS
            },
        }
        formula_result["lopo_direction_summary"] = {}
        for metric in ("mrr_delta", "exam_improvement", "gmar_f_delta"):
            formula_result["lopo_direction_summary"][metric] = {}
            for subset in ("all_paired", "omission_paired"):
                values = [
                    float(group[metric][f"{subset}_mean"])
                    for group in leave_one_out.values()
                ]
                formula_result["lopo_direction_summary"][metric][subset] = {
                    "all_nonnegative": all(value >= 0 for value in values),
                    "maximum": max(values),
                    "minimum": min(values),
                }
        result["by_formula"][formula] = formula_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    compact = {
        formula: {
            "equal_project": details["equal_project_weighted"],
            "lopo_direction": details["lopo_direction_summary"],
            "top10_dominance": details["topk_dominance"]["10"],
        }
        for formula, details in result["by_formula"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
