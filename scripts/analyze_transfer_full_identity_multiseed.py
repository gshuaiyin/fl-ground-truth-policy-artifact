"""Evaluate ground-truth policies on full-identity TRANSFER reruns.

The analysis uses complete ``Class@line`` candidate identities, never the lossy
line-number-only LLMAO export.  Five predeclared deterministic runs are treated
as a robustness family; no best-seed selection is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


BUDGETS = (1, 3, 5, 10)
POLICIES = ("direct", "strict", "broad")
EXPECTED_SEEDS = (888, 1888, 2888, 3888, 4888)
EXCLUDED_FROM_SECONDARY_PUBLIC_EXPORT = {
    "Chart-23", "Closure-28", "Lang-23", "Lang-25", "Lang-29",
    "Lang-36", "Lang-42", "Lang-62", "Math-100", "Math-11",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_sha256(paths: list[Path], root: Path) -> str:
    """Hash names and bytes so the selected annotation corpus is immutable."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def normalize_annotation(raw: str) -> tuple[str, int]:
    path, line, *_ = raw.strip().split("#")
    return path.removesuffix(".java").replace("/", ".").split("$", 1)[0], int(line)


def normalize_identity(identity: str) -> tuple[str, int]:
    name, line = identity.rsplit("@", 1)
    return name.split("$", 1)[0], int(line)


def annotation_sets(directory: Path, bug_id: str):
    direct, markers, candidates, strict_additions = set(), set(), set(), set()
    buggy_path = directory / f"{bug_id}.buggy.lines"
    for raw in buggy_path.read_text(encoding="utf-8", errors="strict").splitlines():
        if not raw.strip():
            continue
        location = normalize_annotation(raw)
        (markers if "FAULT_OF_OMISSION" in raw else direct).add(location)
    candidate_path = directory / f"{bug_id}.candidates"
    if candidate_path.is_file():
        for raw in candidate_path.read_text(encoding="utf-8", errors="strict").splitlines():
            if not raw.strip():
                continue
            left, right = raw.split(",", 1)
            anchor, target = normalize_annotation(left), normalize_annotation(right)
            candidates.add(target)
            if anchor in markers:
                strict_additions.add(target)
    return {
        "direct": direct,
        "strict": direct | strict_additions,
        "broad": direct | candidates,
    }


def read_ranking(path: Path):
    identities, scores = [], []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        header = next(reader)
        if header != ["identity", "score"]:
            raise AssertionError((path, header))
        for row in reader:
            if len(row) != 2:
                raise AssertionError((path, row))
            identities.append(row[0])
            scores.append(float(row[1]))
    if len(identities) != len(set(identities)):
        raise AssertionError((path, "duplicate full identities"))
    if any(scores[i] < scores[i + 1] for i in range(len(scores) - 1)):
        raise AssertionError((path, "not score-descending"))
    average, tie_end = [0.0] * len(scores), [0] * len(scores)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[end] == scores[start]:
            end += 1
        rank = ((start + 1) + end) / 2.0
        for index in range(start, end):
            average[index] = rank
            tie_end[index] = end
        start = end
    return identities, scores, average, tie_end


def random_hit_probability(inspected: int, targets: int, population: int) -> float:
    inspected = min(population, inspected)
    if population - inspected < targets:
        return 1.0
    log_no_hit = (
        math.lgamma(population - inspected + 1)
        - math.lgamma(population - inspected - targets + 1)
        - math.lgamma(population + 1)
        + math.lgamma(population - targets + 1)
    )
    return -math.expm1(log_no_hit)


def gmar_score(inspected: int, targets: int, population: int) -> float:
    return 1.0 - random_hit_probability(inspected, targets, population)


def exact_sign_p(gains: int, losses: int) -> float:
    n = gains + losses
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(gains, losses) + 1)) / 2**n)


def holm_adjust(values: list[float]) -> list[float]:
    adjusted = [1.0] * len(values)
    running = 0.0
    for order, index in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        running = max(running, min(1.0, (len(values) - order) * values[index]))
        adjusted[index] = running
    return adjusted


def bootstrap_mean_ci(values: np.ndarray, seed: int, replicates: int = 20_000):
    if values.size == 0:
        return None
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(20):
        draw = values[rng.integers(0, len(values), size=(replicates // 20, len(values)))]
        samples.append(draw.mean(axis=1))
    return [float(value) for value in np.quantile(np.concatenate(samples), [0.025, 0.975])]


def map_policy(locations, normalized_to_indices, average, tie_end, population):
    mapped = []
    for location in sorted(locations):
        indices = normalized_to_indices.get(location, [])
        if indices:
            best_average = min(average[index] for index in indices)
            best_end = min(tie_end[index] for index in indices if average[index] == best_average)
            mapped.append((location, best_average, best_end))
    if not mapped:
        return {
            "targets": len(locations), "mapped_targets": 0, "mapped_locations": [],
            "rank": None, "tie_end": None, "exam": None, "gmar_f": None, "gmar_a": None,
        }
    first_rank = min(item[1] for item in mapped)
    first_end = min(item[2] for item in mapped if item[1] == first_rank)
    return {
        "targets": len(locations),
        "mapped_targets": len(mapped),
        "mapped_locations": [f"{name}@{line}" for (name, line), _, _ in mapped],
        "rank": float(first_rank),
        "tie_end": int(first_end),
        "exam": float(first_rank / population),
        "gmar_f": float(gmar_score(first_end, len(mapped), population)),
        "gmar_a": float(np.mean([(population - item[2]) / population for item in mapped])),
    }


def policy_metrics(rows, policy: str):
    selected = [row for row in rows if row[policy]["rank"] is not None]
    ranks = np.asarray([row[policy]["rank"] for row in selected], dtype=float)
    return {
        "evaluable": len(selected),
        "mapped_targets_total": sum(row[policy]["mapped_targets"] for row in selected),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "mrr": float(np.mean(1 / ranks)),
        "mean_exam": float(np.mean([row[policy]["exam"] for row in selected])),
        "mean_gmar_f": float(np.mean([row[policy]["gmar_f"] for row in selected])),
        "mean_gmar_a": float(np.mean([row[policy]["gmar_a"] for row in selected])),
        "topk": {str(k): int(np.sum(ranks <= k)) for k in BUDGETS},
    }


def compare(rows, left: str, right: str, seed: int):
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
    for budget in BUDGETS:
        gains = sum(row[left]["rank"] > budget >= row[right]["rank"] for row in paired)
        losses = sum(row[left]["rank"] <= budget < row[right]["rank"] for row in paired)
        result["budgets"][str(budget)] = {
            "left_hits_paired": sum(row[left]["rank"] <= budget for row in paired),
            "right_hits_paired": sum(row[right]["rank"] <= budget for row in paired),
            "gains": gains,
            "losses": losses,
            "p_raw": exact_sign_p(gains, losses),
        }
    arrays = {
        "delta_mrr_right_minus_left": np.asarray([1 / row[right]["rank"] - 1 / row[left]["rank"] for row in paired]),
        "delta_rank_improvement_left_minus_right": np.asarray([row[left]["rank"] - row[right]["rank"] for row in paired]),
        "delta_exam_improvement_left_minus_right": np.asarray([row[left]["exam"] - row[right]["exam"] for row in paired]),
        "delta_gmar_f_right_minus_left": np.asarray([row[right]["gmar_f"] - row[left]["gmar_f"] for row in paired]),
        "delta_gmar_a_right_minus_left": np.asarray([row[right]["gmar_a"] - row[left]["gmar_a"] for row in paired]),
    }
    result["continuous"] = {
        name: {"mean": float(values.mean()), "bootstrap_95ci": bootstrap_mean_ci(values, seed + i)}
        for i, (name, values) in enumerate(arrays.items())
    }
    result["rank_direction"] = {
        "right_better": sum(row[right]["rank"] < row[left]["rank"] for row in paired),
        "equal": sum(row[right]["rank"] == row[left]["rank"] for row in paired),
        "right_worse": sum(row[right]["rank"] > row[left]["rank"] for row in paired),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--statements", type=Path, required=True)
    parser.add_argument("--official-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs, annotations, output = args.runs.resolve(), args.annotations.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    with args.statements.resolve().open("rb") as stream:
        official_statements = pickle.load(stream)
    if len(official_statements) != 395:
        raise AssertionError(len(official_statements))
    official_validation = json.loads(args.official_validation.resolve().read_text(encoding="utf-8"))
    if official_validation.get("status") != "pass" or official_validation.get("exact_versions") != 40:
        raise AssertionError("official fold-1 validation did not pass 40/40")

    included_bug_ids = sorted(
        version.replace("_", "-", 1) for version in official_statements
        if version.replace("_", "-", 1) not in EXCLUDED_FROM_SECONDARY_PUBLIC_EXPORT
    )
    annotation_files = []
    for bug_id in included_bug_ids:
        buggy = annotations / f"{bug_id}.buggy.lines"
        if not buggy.is_file():
            raise FileNotFoundError(buggy)
        annotation_files.append(buggy)
        candidates = annotations / f"{bug_id}.candidates"
        if candidates.is_file():
            annotation_files.append(candidates)

    seed_results = []
    improvement_counts = Counter()
    expected_mapping_signature = None
    paired_bug_ids = None
    for seed in EXPECTED_SEEDS:
        run = runs / f"seed_{seed}"
        if not (run / "COMPLETE").is_file():
            raise FileNotFoundError(run / "COMPLETE")
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        rows = []
        for version in sorted(official_statements):
            bug_id = version.replace("_", "-", 1)
            if bug_id in EXCLUDED_FROM_SECONDARY_PUBLIC_EXPORT:
                continue
            ranking_path = run / "rankings" / version / "ranking.tsv"
            identities, _, average, tie_end = read_ranking(ranking_path)
            if set(identities) != set(official_statements[version]):
                raise AssertionError((seed, version, "ranking identities differ from official statements"))
            normalized_to_indices = defaultdict(list)
            for index, identity in enumerate(identities):
                normalized_to_indices[normalize_identity(identity)].append(index)
            policies = annotation_sets(annotations, bug_id)
            policy_rows = {
                name: map_policy(locations, normalized_to_indices, average, tie_end, len(identities))
                for name, locations in policies.items()
            }
            if not policies["direct"] <= policies["strict"] <= policies["broad"]:
                raise AssertionError((bug_id, "policy nesting"))
            for left, right in (("direct", "strict"), ("strict", "broad")):
                if policy_rows[left]["rank"] is not None and policy_rows[right]["rank"] is not None:
                    if policy_rows[right]["rank"] > policy_rows[left]["rank"]:
                        raise AssertionError((seed, bug_id, left, right))
            row = {"seed": seed, "bug_id": bug_id, "N": len(identities), **policy_rows}
            rows.append(row)
            if policy_rows["direct"]["rank"] is not None and policy_rows["strict"]["rank"] is not None:
                if policy_rows["strict"]["rank"] < policy_rows["direct"]["rank"]:
                    improvement_counts[bug_id] += 1

        mapping_signature = [
            (row["bug_id"], *(row[p]["mapped_targets"] for p in POLICIES)) for row in rows
        ]
        if expected_mapping_signature is None:
            expected_mapping_signature = mapping_signature
        elif mapping_signature != expected_mapping_signature:
            raise AssertionError((seed, "mapping coverage changed across seeds"))
        current_paired_ids = {
            row["bug_id"] for row in rows
            if row["direct"]["rank"] is not None and row["strict"]["rank"] is not None
        }
        if paired_bug_ids is None:
            paired_bug_ids = current_paired_ids
        elif paired_bug_ids != current_paired_ids:
            raise AssertionError((seed, "paired bug set changed across seeds"))
        comparisons = {
            "direct_to_strict": compare(rows, "direct", "strict", 20260916 + seed),
            "direct_to_broad": compare(rows, "direct", "broad", 20261916 + seed),
        }
        tests = [comparisons[name]["budgets"][str(k)] for name in comparisons for k in BUDGETS]
        for test, adjusted in zip(tests, holm_adjust([test["p_raw"] for test in tests])):
            test["p_holm_8"] = adjusted
        seed_result = {
            "seed": seed,
            "run_summary": manifest["summary"],
            "policies": {policy: policy_metrics(rows, policy) for policy in POLICIES},
            "comparisons": comparisons,
            "rows": rows,
        }
        seed_results.append(seed_result)
        print(json.dumps({
            "seed": seed,
            "author_topk": manifest["summary"]["topk"],
            "direct_to_strict": comparisons["direct_to_strict"],
        }, ensure_ascii=False), flush=True)

    cross_seed = {
        "seeds": list(EXPECTED_SEEDS),
        "author_topk": {
            str(k): [item["run_summary"]["topk"][str(k)] for item in seed_results]
            for k in (1, 3, 5, 10)
        },
        "direct_to_strict": {
            "gained_evaluable": [item["comparisons"]["direct_to_strict"]["gained_evaluable"] for item in seed_results],
            "rank_better": [item["comparisons"]["direct_to_strict"]["rank_direction"]["right_better"] for item in seed_results],
            "delta_mrr": [item["comparisons"]["direct_to_strict"]["continuous"]["delta_mrr_right_minus_left"]["mean"] for item in seed_results],
            "topk_gains": {
                str(k): [item["comparisons"]["direct_to_strict"]["budgets"][str(k)]["gains"] for item in seed_results]
                for k in BUDGETS
            },
            "topk_holm_p": {
                str(k): [item["comparisons"]["direct_to_strict"]["budgets"][str(k)]["p_holm_8"] for item in seed_results]
                for k in BUDGETS
            },
        },
        "bug_improvement_frequency": {
            str(times): sum(improvement_counts.get(bug_id, 0) == times for bug_id in paired_bug_ids)
            for times in range(0, len(EXPECTED_SEEDS) + 1)
        },
        "bugs_ever_improved": len(improvement_counts),
        "bugs_improved_all_seeds": sum(count == len(EXPECTED_SEEDS) for count in improvement_counts.values()),
    }
    result = {
        "design": {
            "method": "TRANSFER",
            "ranking_identity": "fully.qualified.Class@line",
            "seeds_predeclared": list(EXPECTED_SEEDS),
            "sample": "385 versions matching the secondary public LLMAO/Transfer-FL export",
            "excluded_versions": sorted(EXCLUDED_FROM_SECONDARY_PUBLIC_EXPORT),
            "primary_comparison": "direct_to_strict",
            "secondary_comparison": "direct_to_broad",
            "budgets": list(BUDGETS),
            "bootstrap_replicates": 20_000,
            "multiplicity": "Holm adjustment across 2 comparisons x 4 budgets within each seed",
        },
        "inputs": {
            "statements_sha256": sha256(args.statements.resolve()),
            "official_fold1_validation_sha256": sha256(args.official_validation.resolve()),
            "official_fold1_validation": {
                "status": official_validation["status"],
                "versions": official_validation["versions"],
                "exact_versions": official_validation["exact_versions"],
            },
            "annotation_files": len(annotation_files),
            "annotation_corpus_sha256": corpus_sha256(annotation_files, annotations),
            "run_manifest_sha256": {
                str(seed): sha256(runs / f"seed_{seed}" / "manifest.json") for seed in EXPECTED_SEEDS
            },
            "analysis_script_sha256": sha256(Path(__file__)),
        },
        "cross_seed": cross_seed,
        "seed_results": seed_results,
    }
    atomic_json(output / "analysis.json", result)
    with (output / "seed_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "author_top1", "author_top3", "author_top5", "author_top10",
                         "direct_evaluable", "strict_evaluable", "gained_evaluable", "rank_better",
                         "top1_gain", "top3_gain", "top5_gain", "top10_gain"])
        for item in seed_results:
            comp = item["comparisons"]["direct_to_strict"]
            writer.writerow([
                item["seed"], *(item["run_summary"]["topk"][str(k)] for k in (1, 3, 5, 10)),
                comp["left_evaluable"], comp["right_evaluable"], comp["gained_evaluable"],
                comp["rank_direction"]["right_better"],
                *(comp["budgets"][str(k)]["gains"] for k in BUDGETS),
            ])
    topk_text = ", ".join(
        f"Top-{k}={cross_seed['author_topk'][str(k)]}" for k in (1, 3, 5, 10)
    )
    primary = cross_seed["direct_to_strict"]
    report = f"""# TRANSFER完整身份五种子复现实验

## 设计

- 排名单位：完整 `fully.qualified.Class@line`，未使用丢失类名的二次JSON。
- 固定种子：{list(EXPECTED_SEEDS)}；全部种子等权报告，不选择最优种子。
- 主比较：direct → strict；样本为与既有学习式实验一致的385个缺陷。
- 官方第1折模型参数交叉验证：40/40个缺陷的故障排名逐项完全一致。

## 结果概览

- 作者故障定义下各随机种子的排名波动：{topk_text}。
- 五次确定性重跑未复现公开的Top-1/3/5总数69/126/145；结合官方代码未固定模型初始化种子，该差异按训练随机性与运行环境敏感性处理，不冒充精确复现。
- direct → strict 新增可评价数：{primary['gained_evaluable']}。
- 配对样本中排名改善数：{primary['rank_better']}。
- Top-1/3/5/10增益：{primary['topk_gains']}。
- 五个种子至少一次改善的缺陷：{cross_seed['bugs_ever_improved']}；五次均改善：{cross_seed['bugs_improved_all_seeds']}。

完整逐缺陷、逐种子结果与置信区间见 `analysis.json`；紧凑表见 `seed_summary.csv`。
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8-sig")
    (output / "COMPLETE").write_text("complete\n", encoding="ascii")
    print(json.dumps(cross_seed, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
