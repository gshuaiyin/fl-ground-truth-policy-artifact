"""Exact null-calibration experiment for GMAR and conventional FL metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gmar import gmar_score, random_hit_probability


def exact_null_summary(population: int, targets: int, top_k: int) -> dict[str, float | int]:
    """Enumerate the exact distribution of the first target in a random ranking."""
    expected_exam = 0.0
    expected_mrr = 0.0
    expected_gmar = 0.0
    gmar_95_false_positive_rate = 0.0
    previous_cdf = 0.0
    probability_sum = 0.0
    for first_rank in range(1, population - targets + 2):
        cdf = random_hit_probability(first_rank, targets, population)
        mass = cdf - previous_cdf
        score = gmar_score(first_rank, targets, population)
        probability_sum += mass
        expected_exam += mass * first_rank / population
        expected_mrr += mass / first_rank
        expected_gmar += mass * score
        if score >= 0.95:
            gmar_95_false_positive_rate += mass
        previous_cdf = cdf
    return {
        "expected_exam": expected_exam,
        "expected_gmar": expected_gmar,
        "expected_mrr": expected_mrr,
        "gmar_ge_0_95_false_positive_rate": gmar_95_false_positive_rate,
        "population": population,
        "probability_sum": probability_sum,
        "random_topk_hit_rate": random_hit_probability(top_k, targets, population),
        "targets": targets,
        "top_k": top_k,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=1000)
    parser.add_argument("--targets", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [exact_null_summary(args.population, m, args.top_k) for m in args.targets]
    output = {
        "design": "exact enumeration; no Monte Carlo error",
        "null_model": "all source-line orderings are equiprobable",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
