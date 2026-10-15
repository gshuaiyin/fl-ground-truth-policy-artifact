"""Ground-truth Multiplicity-Adjusted Rank (GMAR).

GMAR is a chance-calibrated evaluation score for first-hit fault localization.
It accounts for the number of acceptable target source lines and uses the full
inspection set at a suspiciousness-tie boundary.
"""

from __future__ import annotations

import math


def random_hit_probability(inspected: int, targets: int, population: int) -> float:
    """Return P(at least one target occurs in a random size-k inspection set).

    This is the hypergeometric probability
    ``1 - choose(N-k, m) / choose(N, m)`` and is evaluated in log space.
    """
    if population <= 0:
        raise ValueError("population must be positive")
    if targets <= 0 or targets > population:
        raise ValueError("targets must be in [1, population]")
    if inspected < 0:
        raise ValueError("inspected must be non-negative")
    inspected = min(population, inspected)
    if inspected == 0:
        return 0.0
    if population - inspected < targets:
        return 1.0
    log_no_hit = (
        math.lgamma(population - inspected + 1)
        - math.lgamma(targets + 1)
        - math.lgamma(population - inspected - targets + 1)
        - math.lgamma(population + 1)
        + math.lgamma(targets + 1)
        + math.lgamma(population - targets + 1)
    )
    return -math.expm1(log_no_hit)


def gmar_score(inspected: int, targets: int, population: int) -> float:
    """Return the probability that random ranking would do strictly worse."""
    return 1.0 - random_hit_probability(inspected, targets, population)
