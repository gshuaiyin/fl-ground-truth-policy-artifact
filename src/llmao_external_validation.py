"""Independent ground-truth sensitivity analysis of public LLMAO rankings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_groundtruth_sensitivity import exact_sign_p, holm_adjust
from gmar import gmar_score
from revision_v140 import annotation_sets


SEED = 20260911
BOOTSTRAP = 20_000
BUDGETS = (1, 3, 5, 10)
POLICIES = ("direct", "strict", "broad")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def outer_class(name: str) -> str:
    return name.removesuffix(".java").replace("/", ".").split("$", 1)[0]


def ranks_desc(scores: dict[int, float]) -> tuple[dict[int, float], dict[int, int]]:
    """Return average and tie-end ranks after numerical descending sorting."""
    ordered = sorted(scores, key=lambda line: (-scores[line], line))
    average: dict[int, float] = {}
    tie_end: dict[int, int] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and scores[ordered[end]] == scores[ordered[start]]:
            end += 1
        avg = ((start + 1) + end) / 2.0
        for line in ordered[start:end]:
            average[line] = avg
            tie_end[line] = end
        start = end
    return average, tie_end


def official_order_rank(raw_scores: dict[str, float], targets: set[int]) -> int | None:
    """Exactly reproduce top_score_window.py, including score-key collisions."""
    score_to_line: dict[float, str] = {}
    for line, score in raw_scores.items():
        score_to_line[score] = line
    line_to_score = {line: score for score, line in score_to_line.items()}
    for index, line in enumerate(line_to_score, 1):
        if int(line) in targets:
            return index
    return None


def policy_metrics(rows: list[dict], policy: str) -> dict:
    selected = [row for row in rows if row[policy]["rank"] is not None]
    if not selected:
        return {"evaluable": 0}
    ranks = np.asarray([row[policy]["rank"] for row in selected], dtype=float)
    ends = np.asarray([row[policy]["tie_end"] for row in selected], dtype=float)
    return {
        "evaluable": len(selected),
        "mapped_targets_total": sum(row[policy]["mapped_targets"] for row in selected),
        "mapped_targets_mean": float(np.mean([row[policy]["mapped_targets"] for row in selected])),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_exam": float(np.mean([row[policy]["exam"] for row in selected])),
        "mean_tie_end": float(ends.mean()),
        "mean_gmar_f": float(np.mean([row[policy]["gmar_f"] for row in selected])),
        "mean_gmar_a": float(np.mean([row[policy]["gmar_a"] for row in selected])),
        "topk": {str(k): int(np.sum(ranks <= k)) for k in BUDGETS},
    }


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> list[float] | None:
    if values.size == 0:
        return None
    rng = np.random.default_rng(seed)
    chunks = []
    for _ in range(20):
        sampled = values[rng.integers(0, len(values), size=(BOOTSTRAP // 20, len(values)))]
        chunks.append(sampled.mean(axis=1))
    return [float(x) for x in np.quantile(np.concatenate(chunks), [0.025, 0.975])]


def compare(rows: list[dict], left: str, right: str, seed: int) -> dict:
    paired = [row for row in rows if row[left]["rank"] is not None and row[right]["rank"] is not None]
    result = {
        "left": left,
        "right": right,
        "left_evaluable": sum(row[left]["rank"] is not None for row in rows),
        "right_evaluable": sum(row[right]["rank"] is not None for row in rows),
        "paired": len(paired),
        "gained_evaluable": sum(row[left]["rank"] is None and row[right]["rank"] is not None for row in rows),
        "lost_evaluable": sum(row[left]["rank"] is not None and row[right]["rank"] is None for row in rows),
        "budgets": {},
    }
    for k in BUDGETS:
        gains = sum(row[left]["rank"] > k and row[right]["rank"] <= k for row in paired)
        losses = sum(row[left]["rank"] <= k and row[right]["rank"] > k for row in paired)
        result["budgets"][str(k)] = {
            "left_hits_paired": sum(row[left]["rank"] <= k for row in paired),
            "right_hits_paired": sum(row[right]["rank"] <= k for row in paired),
            "gains": gains,
            "losses": losses,
            "p_raw": exact_sign_p(gains, losses),
        }
    definitions = {
        "delta_mrr_right_minus_left": np.asarray([1 / row[right]["rank"] - 1 / row[left]["rank"] for row in paired]),
        "delta_rank_improvement_left_minus_right": np.asarray([row[left]["rank"] - row[right]["rank"] for row in paired]),
        "delta_exam_improvement_left_minus_right": np.asarray([row[left]["exam"] - row[right]["exam"] for row in paired]),
        "delta_gmar_f_right_minus_left": np.asarray([row[right]["gmar_f"] - row[left]["gmar_f"] for row in paired]),
        "delta_gmar_a_right_minus_left": np.asarray([row[right]["gmar_a"] - row[left]["gmar_a"] for row in paired]),
    }
    result["continuous"] = {
        name: {"mean": float(values.mean()), "bootstrap_95ci": bootstrap_mean_ci(values, seed + index)}
        for index, (name, values) in enumerate(definitions.items())
    }
    result["rank_direction"] = {
        "right_better": sum(row[right]["rank"] < row[left]["rank"] for row in paired),
        "equal": sum(row[right]["rank"] == row[left]["rank"] for row in paired),
        "right_worse": sum(row[right]["rank"] > row[left]["rank"] for row in paired),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("results/llmao_external_validation"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    analysis_path = output / "analysis.json"
    if analysis_path.exists():
        raise FileExistsError(f"Refusing to overwrite {analysis_path}")

    llmao_root = root / "external/llmao_public"
    score_root = llmao_root / "score_llmao_window"
    annotations = root / "external/fault-localization-data/rjust-fault-localization-data-db9a70bfeaef/analysis/pipeline-scripts/buggy-lines"
    legacy_path = root / "results/full/groundtruth_sensitivity.json"
    legacy_ids = {row["bug_id"] for row in json.loads(legacy_path.read_text(encoding="utf-8"))["rows"]}
    commit = __import__("subprocess").check_output(
        ["git", "-C", str(llmao_root), "rev-parse", "HEAD"], text=True
    ).strip()

    rows: list[dict] = []
    manifest: dict[str, object] = {
        "llmao_commit": commit,
        "protocol_sha256": sha256(output / "PROTOCOL.md"),
        "analysis_script_sha256": sha256(Path(__file__)),
        "legacy_index_sha256": sha256(legacy_path),
        "inputs": {},
    }
    metadata_files = sorted(score_root.glob("*/*/metadata.json"), key=lambda p: (p.parent.parent.name, int(p.parent.name)))
    for metadata_path in metadata_files:
        sus_path = metadata_path.with_name("sus.json")
        if not sus_path.is_file():
            raise FileNotFoundError(sus_path)
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw_scores = json.loads(sus_path.read_text(encoding="utf-8"))
        project, bug = str(meta["project"]), str(meta["bug_id"])
        bug_id = f"{project}-{bug}"
        if bug_id != f"{metadata_path.parent.parent.name}-{metadata_path.parent.name}":
            raise AssertionError((bug_id, metadata_path))
        if not raw_scores:
            raise AssertionError((bug_id, "empty scores"))
        scores = {int(line): float(score) for line, score in raw_scores.items()}
        if len(scores) != len(raw_scores) or any(line <= 0 for line in scores):
            raise AssertionError((bug_id, "invalid line keys"))
        if any(not math.isfinite(score) for score in scores.values()):
            raise AssertionError((bug_id, "non-finite score"))
        average, tie_end = ranks_desc(scores)
        klass = outer_class(str(meta["classpath"]))
        direct, markers, candidates, strict_additions, relations = annotation_sets(annotations, bug_id)
        target_sets = {
            "direct": direct,
            "strict": direct | strict_additions,
            "broad": direct | candidates,
        }
        policy_data = {}
        for policy, targets in target_sets.items():
            same_class = {line for name, line in targets if outer_class(name) == klass}
            mapped = sorted(same_class & scores.keys())
            if mapped:
                first_rank = min(average[line] for line in mapped)
                first_end = min(tie_end[line] for line in mapped if average[line] == first_rank)
                policy_data[policy] = {
                    "targets_all_classes": len(targets),
                    "targets_same_class": len(same_class),
                    "mapped_targets": len(mapped),
                    "mapped_lines": mapped,
                    "rank": float(first_rank),
                    "tie_end": int(first_end),
                    "exam": float(first_rank / len(scores)),
                    "gmar_f": float(gmar_score(first_end, len(mapped), len(scores))),
                    "gmar_a": float(np.mean([(len(scores) - tie_end[line]) / len(scores) for line in mapped])),
                }
            else:
                policy_data[policy] = {
                    "targets_all_classes": len(targets),
                    "targets_same_class": len(same_class),
                    "mapped_targets": 0,
                    "mapped_lines": [],
                    "rank": None,
                    "tie_end": None,
                    "exam": None,
                    "gmar_f": None,
                    "gmar_a": None,
                }
        own_targets = {int(line) for line in meta["bug_line_number"]}
        own_mapped = own_targets & scores.keys()
        own_numeric_rank = min((average[line] for line in own_mapped), default=None)
        own_official_rank = official_order_rank(raw_scores, own_targets)
        numerical_order = sorted(scores, key=lambda line: (-scores[line], line))
        json_order = [int(line) for line in raw_scores]
        row = {
            "bug_id": bug_id,
            "project": project,
            "classpath": klass,
            "N_reported": len(scores),
            "metadata_num_lines": int(meta["num_lines"]),
            "unique_score_values": len(set(scores.values())),
            "json_order_is_numeric_desc": json_order == numerical_order,
            "own_targets": sorted(own_targets),
            "own_mapped_targets": sorted(own_mapped),
            "own_official_order_rank": own_official_rank,
            "own_numeric_average_rank": own_numeric_rank,
            "has_omission_marker": bool(markers),
            "has_candidate_file": (annotations / f"{bug_id}.candidates").is_file(),
            "strict_relations": sum(anchor in markers for anchor, _ in relations),
            **policy_data,
        }
        for narrower, wider in (("direct", "strict"), ("strict", "broad")):
            if row[narrower]["rank"] is not None and row[wider]["rank"] is not None:
                assert row[wider]["rank"] <= row[narrower]["rank"]
        rows.append(row)
        for path in (metadata_path, sus_path):
            manifest["inputs"][str(path.relative_to(root))] = sha256(path)

    observed_ids = {row["bug_id"] for row in rows}
    if len(rows) != len(observed_ids):
        raise AssertionError("duplicate bug IDs")
    official_counts = {
        str(k): sum(row["own_official_order_rank"] is not None and row["own_official_order_rank"] <= k for row in rows)
        for k in (1, 3, 5)
    }
    if official_counts != {"1": 24, "3": 52, "5": 77}:
        raise AssertionError(("README reproduction failed", official_counts))
    numeric_counts = {
        str(k): sum(row["own_numeric_average_rank"] is not None and row["own_numeric_average_rank"] <= k for row in rows)
        for k in BUDGETS
    }
    comparisons = {
        "direct_to_strict": compare(rows, "direct", "strict", SEED + 100),
        "direct_to_broad": compare(rows, "direct", "broad", SEED + 200),
    }
    budget_tests = [
        comparisons[name]["budgets"][str(k)]
        for name in ("direct_to_strict", "direct_to_broad")
        for k in BUDGETS
    ]
    for test, adjusted in zip(budget_tests, holm_adjust([test["p_raw"] for test in budget_tests])):
        test["p_holm_8"] = adjusted

    result = {
        "design": {
            "status": "post_hoc_extension_with_protocol_frozen_before_policy_recalculation",
            "seed": SEED,
            "bootstrap_replicates": BOOTSTRAP,
            "budgets": list(BUDGETS),
            "candidate_universe": "lines actually present in public sus.json",
            "primary_comparison": "direct_to_strict",
            "secondary_comparison": "direct_to_broad",
        },
        "scope": {
            "public_rankings": len(rows),
            "legacy_defects4j_bugs": len(legacy_ids),
            "missing_public_bug_ids": sorted(legacy_ids - observed_ids),
            "unexpected_public_bug_ids": sorted(observed_ids - legacy_ids),
            "projects": dict(Counter(row["project"] for row in rows)),
            "reported_N_min_median_max": [
                min(row["N_reported"] for row in rows),
                float(np.median([row["N_reported"] for row in rows])),
                max(row["N_reported"] for row in rows),
            ],
            "json_order_not_numeric_desc": sum(not row["json_order_is_numeric_desc"] for row in rows),
            "score_tie_bugs": sum(row["unique_score_values"] < row["N_reported"] for row in rows),
            "omission_marker_bugs": sum(row["has_omission_marker"] for row in rows),
            "strict_candidate_relation_bugs": sum(row["strict_relations"] > 0 for row in rows),
        },
        "author_result_reproduction": {
            "official_order_topk": official_counts,
            "expected_readme_topk": {"1": 24, "3": 52, "5": 77},
            "numeric_average_rank_topk": numeric_counts,
            "own_truth_evaluable_numeric": sum(row["own_numeric_average_rank"] is not None for row in rows),
        },
        "policies": {policy: policy_metrics(rows, policy) for policy in POLICIES},
        "comparisons": comparisons,
        "mapping": {
            policy: {
                "targets_all_classes": sum(row[policy]["targets_all_classes"] for row in rows),
                "targets_same_selected_class": sum(row[policy]["targets_same_class"] for row in rows),
                "targets_in_reported_universe": sum(row[policy]["mapped_targets"] for row in rows),
                "bugs_with_same_class_targets": sum(row[policy]["targets_same_class"] > 0 for row in rows),
                "bugs_with_reported_targets": sum(row[policy]["mapped_targets"] > 0 for row in rows),
            }
            for policy in POLICIES
        },
        "checks": {
            "readme_exactly_reproduced": True,
            "all_json_valid": True,
            "nested_rank_invariants_passed": True,
            "input_files": len(manifest["inputs"]),
        },
        "rows": rows,
        "manifest": manifest,
    }
    analysis_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (output / "per_bug.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["bug_id", "project", "classpath", "N_reported", "own_numeric_rank", *[f"{p}_mapped" for p in POLICIES], *[f"{p}_rank" for p in POLICIES]])
        for row in rows:
            writer.writerow([row["bug_id"], row["project"], row["classpath"], row["N_reported"], row["own_numeric_average_rank"], *[row[p]["mapped_targets"] for p in POLICIES], *[row[p]["rank"] for p in POLICIES]])

    print(json.dumps({
        "finished": True,
        "scope": result["scope"],
        "author_result_reproduction": result["author_result_reproduction"],
        "policies": result["policies"],
        "comparisons": result["comparisons"],
        "mapping": result["mapping"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
