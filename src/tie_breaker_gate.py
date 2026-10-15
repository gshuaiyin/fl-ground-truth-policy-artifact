from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spectrum import (
    average_ranks_desc,
    deleted_source_locations,
    exact_top_k,
    load_gzoltar,
    statement_matches_location,
    suspiciousness,
)


DEVELOPMENT_PROJECTS = {"Chart", "Closure", "Math"}
HOLDOUT_PROJECTS = {"Lang", "Mockito", "Time"}


def focused_test_weighted_ochiai(coverage: np.ndarray, failed: np.ndarray) -> np.ndarray:
    """Ochiai with focused tests weighted more heavily.

    A test's raw weight is the inverse square root of the number of statements
    it covers. Weights are normalized to mean one separately for passing and
    failing tests, preventing class size from changing the basic Ochiai scale.
    """
    row_size = coverage.sum(axis=1, dtype=np.float64)
    weights = 1.0 / np.sqrt(np.maximum(row_size, 1.0))
    for mask in (failed, ~failed):
        if mask.any():
            weights[mask] *= float(mask.sum()) / weights[mask].sum()
    failing_weights = weights[failed]
    passing_weights = weights[~failed]
    ef = (coverage[failed] * failing_weights[:, None]).sum(axis=0)
    ep = (coverage[~failed] * passing_weights[:, None]).sum(axis=0)
    total_failing = failing_weights.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        score = ef / np.sqrt(total_failing * (ef + ep))
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def lexicographic_average_ranks_desc(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Average ranks for ties on both primary and secondary scores."""
    primary = np.asarray(primary, dtype=float)
    secondary = np.asarray(secondary, dtype=float)
    order = np.lexsort((-secondary, -primary))
    ranks = np.empty(primary.size, dtype=float)
    start = 0
    while start < primary.size:
        end = start + 1
        while (
            end < primary.size
            and primary[order[end]] == primary[order[start]]
            and secondary[order[end]] == secondary[order[start]]
        ):
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end
    return ranks


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, subset in (
        ("development", [r for r in rows if r["project"] in DEVELOPMENT_PROJECTS]),
        ("holdout", [r for r in rows if r["project"] in HOLDOUT_PROJECTS]),
        ("all", rows),
    ):
        base = np.asarray([float(r["baseline_rank"]) for r in subset])
        method = np.asarray([float(r["method_rank"]) for r in subset])
        split: dict[str, object] = {
            "n": len(subset),
            "mean_exam_baseline": float(np.mean([float(r["baseline_exam"]) for r in subset])),
            "mean_exam_method": float(np.mean([float(r["method_exam"]) for r in subset])),
            "median_rank_baseline": float(np.median(base)),
            "median_rank_method": float(np.median(method)),
        }
        for k in (1, 5, 10, 20):
            before = base <= k
            after = method <= k
            split[f"top{k}"] = {
                "baseline": int(before.sum()),
                "method": int(after.sum()),
                "gained": int((~before & after).sum()),
                "lost": int((before & ~after).sum()),
                "net": int(after.sum() - before.sum()),
            }
        result[name] = split

    per_project = {}
    for project in sorted({str(r["project"]) for r in rows}):
        subset = [r for r in rows if r["project"] == project]
        base = np.asarray([float(r["baseline_rank"]) for r in subset])
        method = np.asarray([float(r["method_rank"]) for r in subset])
        per_project[project] = {
            "n": len(subset),
            "top10_baseline": int((base <= 10).sum()),
            "top10_method": int((method <= 10).sum()),
            "top10_net": int((method <= 10).sum() - (base <= 10).sum()),
            "mean_rank_change": float(np.mean(method - base)),
        }
    result["per_project"] = per_project
    holdout = result["holdout"]
    top10_relative_gain = holdout["top10"]["net"] / max(1, holdout["top10"]["baseline"])
    exam_relative_gain = (
        (holdout["mean_exam_baseline"] - holdout["mean_exam_method"])
        / holdout["mean_exam_baseline"]
        if holdout["mean_exam_baseline"] else 0.0
    )
    result["gate"] = {
        "holdout_top10_relative_improvement_at_least_5pct": bool(
            top10_relative_gain >= 0.05
        ),
        "holdout_mean_exam_relative_improvement_at_least_1pct": bool(
            exam_relative_gain >= 0.01
        ),
        "holdout_top20_not_worse": bool(holdout["top20"]["net"] >= 0),
        "holdout_no_more_than_one_top10_loss": bool(holdout["top10"]["lost"] <= 1),
    }
    result["gate"]["pass"] = bool(all(result["gate"].values()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spectra_root", type=Path)
    parser.add_argument("patch_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []
    for project_dir in sorted(path for path in args.spectra_root.iterdir() if path.is_dir()):
        for bug_dir in sorted(
            (path for path in project_dir.iterdir() if path.is_dir()),
            key=lambda path: int(path.name),
        ):
            project = project_dir.name
            patch = args.patch_root / project / f"{bug_dir.name}.src.patch"
            try:
                spectrum = load_gzoltar(bug_dir)
                locations = deleted_source_locations(patch)
                faulty = np.asarray(
                    [
                        index
                        for index, statement in enumerate(spectrum.statements)
                        if any(
                            statement_matches_location(statement, location)
                            for location in locations
                        )
                    ],
                    dtype=int,
                )
                if faulty.size == 0:
                    excluded.append(
                        {"project": project, "bug": bug_dir.name, "reason": "no mapped faulty probe"}
                    )
                    continue
                baseline_score = suspiciousness(spectrum.coverage, spectrum.failed, "ochiai")
                secondary_score = focused_test_weighted_ochiai(
                    spectrum.coverage, spectrum.failed
                )
                baseline_rank = float(np.min(average_ranks_desc(baseline_score)[faulty]))
                method_rank = float(
                    np.min(
                        lexicographic_average_ranks_desc(
                            baseline_score, secondary_score
                        )[faulty]
                    )
                )
                kth = exact_top_k(
                    baseline_score, spectrum.statements, min(10, baseline_score.size)
                )[-1]
                rows.append(
                    {
                        "project": project,
                        "bug": bug_dir.name,
                        "statements": len(spectrum.statements),
                        "baseline_rank": baseline_rank,
                        "method_rank": method_rank,
                        "baseline_exam": baseline_rank / len(spectrum.statements),
                        "method_exam": method_rank / len(spectrum.statements),
                        "top10_boundary_tie_size": int(
                            (baseline_score == baseline_score[kth]).sum()
                        ),
                    }
                )
            except Exception as error:
                excluded.append(
                    {
                        "project": project,
                        "bug": bug_dir.name,
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )

    summary = summarize(rows)
    payload = {"summary": summary, "bugs": rows, "excluded": excluded}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
