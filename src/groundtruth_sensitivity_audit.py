"""Measure how omission-aware ground truth changes SBFL conclusions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gmar import gmar_score
from spectrum import (
    average_ranks_desc,
    deleted_source_locations,
    load_gzoltar,
    suspiciousness,
)


def normalized_location(raw: str) -> tuple[str, int]:
    path, line, *_ = raw.strip().split("#")
    return path.removesuffix(".java").replace("/", "."), int(line)


def official_locations(
    annotations: Path, project: str, bug: str
) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    direct: set[tuple[str, int]] = set()
    omission: set[tuple[str, int]] = set()
    buggy_lines = annotations / f"{project}-{bug}.buggy.lines"
    for raw in buggy_lines.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.strip() and "FAULT_OF_OMISSION" not in raw:
            direct.add(normalized_location(raw))
    candidates = annotations / f"{project}-{bug}.candidates"
    if candidates.is_file():
        for raw in candidates.read_text(encoding="utf-8", errors="replace").splitlines():
            if "," in raw:
                omission.add(normalized_location(raw.split(",", 1)[1]))
    return direct, omission


def mapped_indices(
    statements: tuple[str, ...], locations: set[tuple[str, int]]
) -> np.ndarray:
    mapped: list[int] = []
    for index, statement in enumerate(statements):
        name, line = statement.rsplit("#", 1)
        base_name = name.split("$", 1)[0]
        if (base_name, int(line)) in locations:
            mapped.append(index)
    return np.asarray(mapped, dtype=int)


def source_line_index(
    statements: tuple[str, ...],
) -> tuple[tuple[tuple[str, int], ...], np.ndarray]:
    """Map GZoltar probes to unique outer-class source lines."""
    locations: list[tuple[str, int]] = []
    indices: dict[tuple[str, int], int] = {}
    probe_to_line: list[int] = []
    for statement in statements:
        name, line = statement.rsplit("#", 1)
        location = (name.split("$", 1)[0], int(line))
        if location not in indices:
            indices[location] = len(locations)
            locations.append(location)
        probe_to_line.append(indices[location])
    return tuple(locations), np.asarray(probe_to_line, dtype=int)


def mapped_line_indices(
    source_lines: tuple[tuple[str, int], ...], locations: set[tuple[str, int]]
) -> np.ndarray:
    return np.asarray(
        [index for index, location in enumerate(source_lines) if location in locations],
        dtype=int,
    )


def summarize(rows: list[dict[str, object]], policy: str, formula: str) -> dict[str, object]:
    key = f"{policy}_{formula}_rank"
    ranks = np.asarray([float(row[key]) for row in rows if row[key] is not None])
    return {
        "evaluable": int(ranks.size),
        "mean_rank": float(ranks.mean()) if ranks.size else None,
        "median_rank": float(np.median(ranks)) if ranks.size else None,
        "mrr": float(np.mean(1.0 / ranks)) if ranks.size else None,
        "top1": int(np.sum(ranks <= 1)),
        "top5": int(np.sum(ranks <= 5)),
        "top10": int(np.sum(ranks <= 10)),
        "top20": int(np.sum(ranks <= 20)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spectra_root", type=Path)
    parser.add_argument("patch_root", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    formulas = ("ochiai", "dstar", "tarantula", "barinel")
    rows: list[dict[str, object]] = []
    for project_dir in sorted(path for path in args.spectra_root.iterdir() if path.is_dir()):
        for bug_dir in sorted(
            (path for path in project_dir.iterdir() if path.is_dir()),
            key=lambda path: int(path.name),
        ):
            project, bug = project_dir.name, bug_dir.name
            spectrum = load_gzoltar(bug_dir)
            deleted = {
                normalized_location(location)
                for location in deleted_source_locations(
                    args.patch_root / project / f"{bug}.src.patch"
                )
            }
            direct, omission = official_locations(args.annotations, project, bug)
            source_lines, probe_to_line = source_line_index(spectrum.statements)
            policies = {
                "deleted_only": mapped_indices(spectrum.statements, deleted),
                "official_direct": mapped_indices(spectrum.statements, direct),
                "official_omission_aware": mapped_indices(
                    spectrum.statements, direct | omission
                ),
            }
            line_policies = {
                "deleted_only": mapped_line_indices(source_lines, deleted),
                "official_direct": mapped_line_indices(source_lines, direct),
                "official_omission_aware": mapped_line_indices(
                    source_lines, direct | omission
                ),
            }
            row: dict[str, object] = {
                "bug_id": f"{project}-{bug}",
                "has_omission_candidates": bool(omission),
                "mapped_deleted_probes": int(policies["deleted_only"].size),
                "mapped_direct_probes": int(policies["official_direct"].size),
                "mapped_official_probes": int(
                    policies["official_omission_aware"].size
                ),
                "mapped_deleted_lines": int(line_policies["deleted_only"].size),
                "mapped_direct_lines": int(line_policies["official_direct"].size),
                "mapped_official_lines": int(
                    line_policies["official_omission_aware"].size
                ),
                "project": project,
                "ranking_size": len(spectrum.statements),
                "line_ranking_size": len(source_lines),
            }
            for formula in formulas:
                formula_scores = suspiciousness(
                    spectrum.coverage, spectrum.failed, formula
                )
                ranks = average_ranks_desc(formula_scores)
                line_scores = np.full(len(source_lines), -np.inf)
                np.maximum.at(line_scores, probe_to_line, formula_scores)
                line_ranks = average_ranks_desc(line_scores)
                for policy, indices in policies.items():
                    row[f"{policy}_{formula}_rank"] = (
                        float(ranks[indices].min()) if indices.size else None
                    )
                    row[f"{policy}_{formula}_tie_expanded_size"] = (
                        int(
                            np.sum(
                                formula_scores >= np.max(formula_scores[indices])
                            )
                        )
                        if indices.size
                        else None
                    )
                for policy, indices in line_policies.items():
                    row[f"line_{policy}_{formula}_rank"] = (
                        float(line_ranks[indices].min()) if indices.size else None
                    )
                    row[f"line_{policy}_{formula}_tie_expanded_size"] = (
                        int(np.sum(line_scores >= np.max(line_scores[indices])))
                        if indices.size
                        else None
                    )
                    if policy == "official_omission_aware":
                        row[f"line_{policy}_{formula}_gmar_average"] = (
                            float(
                                np.mean(
                                    [
                                        gmar_score(
                                            int(np.sum(line_scores >= line_scores[index])),
                                            1,
                                            len(source_lines),
                                        )
                                        for index in indices
                                    ]
                                )
                            )
                            if indices.size
                            else None
                        )
            rows.append(row)

    summary = {
        policy: {formula: summarize(rows, policy, formula) for formula in formulas}
        for policy in ("deleted_only", "official_direct", "official_omission_aware")
    }
    comparable = [
        row
        for row in rows
        if row["deleted_only_ochiai_rank"] is not None
        and row["official_omission_aware_ochiai_rank"] is not None
    ]
    deltas = np.asarray(
        [
            float(row["deleted_only_ochiai_rank"])
            - float(row["official_omission_aware_ochiai_rank"])
            for row in comparable
        ]
    )
    output = {
        "n_bugs": len(rows),
        "ochiai_paired": {
            "comparable_bugs": len(comparable),
            "deleted_rank_better": int(np.sum(deltas < 0)),
            "equal": int(np.sum(deltas == 0)),
            "median_deleted_minus_official_rank": float(np.median(deltas)),
            "official_rank_better": int(np.sum(deltas > 0)),
        },
        "rows": rows,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"n_bugs": len(rows), "ochiai_paired": output["ochiai_paired"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
