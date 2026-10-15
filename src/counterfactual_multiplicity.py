"""Within-bug exact counterfactual experiment for ground-truth multiplicity."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from analyze_groundtruth_sensitivity import interval
from gmar import gmar_score
from groundtruth_sensitivity_audit import (
    mapped_line_indices,
    official_locations,
    source_line_index,
)
from spectrum import average_ranks_desc, load_gzoltar, suspiciousness


SUBSET_SIZES = (1, 2, 4)
METRICS = ("top10", "mrr", "exam", "gmar")
REPORT_METRICS = METRICS + ("gmar_average",)


def choose(n: int, k: int) -> int:
    return math.comb(n, k) if n >= k else 0


def exact_subset_expectations(
    target_scores: np.ndarray,
    line_scores: np.ndarray,
    line_ranks: np.ndarray,
    subset_size: int,
) -> dict[str, float]:
    """Average metrics over every equally likely size-q target subset."""
    target_count = int(target_scores.size)
    denominator = choose(target_count, subset_size)
    if denominator == 0:
        raise ValueError("subset is larger than target set")
    expected = {metric: 0.0 for metric in METRICS}
    better_targets = 0
    for score in sorted(set(float(value) for value in target_scores), reverse=True):
        in_group = int(np.sum(target_scores == score))
        mass = (
            choose(target_count - better_targets, subset_size)
            - choose(target_count - better_targets - in_group, subset_size)
        ) / denominator
        if mass:
            tied_lines = np.flatnonzero(line_scores == score)
            rank = float(line_ranks[tied_lines[0]])
            tie_end = int(np.sum(line_scores >= score))
            expected["top10"] += mass * (rank <= 10)
            expected["mrr"] += mass / rank
            expected["exam"] += mass * rank / line_scores.size
            expected["gmar"] += mass * gmar_score(
                tie_end, subset_size, int(line_scores.size)
            )
        better_targets += in_group
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spectra_root", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for project_dir in sorted(path for path in args.spectra_root.iterdir() if path.is_dir()):
        for bug_dir in sorted(
            (path for path in project_dir.iterdir() if path.is_dir()),
            key=lambda path: int(path.name),
        ):
            project, bug = project_dir.name, bug_dir.name
            spectrum = load_gzoltar(bug_dir)
            direct, omission = official_locations(args.annotations, project, bug)
            source_lines, probe_to_line = source_line_index(spectrum.statements)
            targets = mapped_line_indices(source_lines, direct | omission)
            if targets.size < max(SUBSET_SIZES):
                continue
            probe_scores = suspiciousness(spectrum.coverage, spectrum.failed, "ochiai")
            line_scores = np.full(len(source_lines), -np.inf)
            np.maximum.at(line_scores, probe_to_line, probe_scores)
            line_ranks = average_ranks_desc(line_scores)
            row: dict[str, object] = {
                "bug_id": f"{project}-{bug}",
                "mapped_targets": int(targets.size),
                "project": project,
                "ranking_size": len(source_lines),
            }
            for subset_size in SUBSET_SIZES:
                values = exact_subset_expectations(
                    line_scores[targets], line_scores, line_ranks, subset_size
                )
                for metric, value in values.items():
                    row[f"m{subset_size}_{metric}"] = value
            # Expected average single-target GMAR is cardinality-stable under
            # uniform target subsampling by linearity of expectation.
            for subset_size in SUBSET_SIZES:
                row[f"m{subset_size}_gmar_average"] = row["m1_gmar"]
            rows.append(row)

    means = {
        f"m{subset_size}": {
            metric: statistics.mean(float(row[f"m{subset_size}_{metric}"]) for row in rows)
            for metric in REPORT_METRICS
        }
        for subset_size in SUBSET_SIZES
    }
    contrasts = {}
    for metric_index, metric in enumerate(REPORT_METRICS):
        deltas = [float(row[f"m4_{metric}"]) - float(row[f"m1_{metric}"]) for row in rows]
        contrasts[metric] = {
            "bug_bootstrap_95ci": interval(deltas, 20261400 + metric_index),
            "mean_m4_minus_m1": statistics.mean(deltas),
        }
    output = {
        "analysis": "exact expectation over all equally likely target subsets",
        "contrasts": contrasts,
        "means": means,
        "n_bugs_with_at_least_4_mapped_targets": len(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
