"""Small, dependency-light utilities for GZoltar spectra.

The pilot intentionally depends only on NumPy so it can run on the current
machine. Confirmatory analysis may add pinned statistical dependencies later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class Spectrum:
    coverage: np.ndarray
    failed: np.ndarray
    statements: tuple[str, ...]


def load_gzoltar(directory: Path) -> Spectrum:
    matrix_path = directory / "matrix"
    spectra_path = directory / "spectra"
    statements = tuple(
        line.strip() for line in spectra_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    rows: list[np.ndarray] = []
    failed: list[bool] = []
    with matrix_path.open("r", encoding="utf-8") as handle:
        for number, raw in enumerate(handle, 1):
            tokens = raw.split()
            if not tokens:
                continue
            outcome = tokens[-1]
            if outcome not in {"+", "-"}:
                raise ValueError(f"Unexpected outcome on matrix line {number}: {outcome}")
            row = np.fromiter((token == "1" for token in tokens[:-1]), dtype=np.bool_)
            if row.size != len(statements):
                raise ValueError(
                    f"Line {number} has {row.size} probes; spectra has {len(statements)}"
                )
            rows.append(row)
            failed.append(outcome == "-")
    if not rows:
        raise ValueError(f"Empty matrix: {matrix_path}")
    return Spectrum(np.vstack(rows), np.asarray(failed, dtype=np.bool_), statements)


def suspiciousness(coverage: np.ndarray, failed: np.ndarray, formula: str) -> np.ndarray:
    failed = np.asarray(failed, dtype=np.bool_)
    passed = ~failed
    ef = coverage[failed].sum(axis=0, dtype=np.float64)
    ep = coverage[passed].sum(axis=0, dtype=np.float64)
    nf = float(failed.sum()) - ef
    npass = float(passed.sum()) - ep
    with np.errstate(divide="ignore", invalid="ignore"):
        if formula == "ochiai":
            score = ef / np.sqrt((ef + nf) * (ef + ep))
        elif formula == "dstar":
            score = ef**2 / (ep + nf)
        elif formula == "tarantula":
            fail_rate = ef / (ef + nf)
            pass_rate = ep / (ep + npass)
            score = fail_rate / (fail_rate + pass_rate)
        elif formula == "barinel":
            score = 1.0 - ep / (ep + ef)
        else:
            raise ValueError(f"Unknown formula: {formula}")
    # DStar legitimately yields +inf when no passing test covers a statement
    # and every failing test does. Preserve that ordering without injecting the
    # maximum float, which would overflow downstream standardization/models.
    finite = score[np.isfinite(score)]
    positive_infinity_cap = float(finite.max() + 1.0) if finite.size else 1.0
    score = np.nan_to_num(
        score, nan=0.0, posinf=positive_infinity_cap, neginf=0.0
    )
    return score


def average_ranks_desc(scores: np.ndarray) -> np.ndarray:
    """One-based descending ranks with the average rank assigned to ties."""
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end
    return ranks


def exact_top_k(scores: np.ndarray, statements: tuple[str, ...], k: int) -> np.ndarray:
    """Deterministic Top-k for set stability; score ties use statement identity."""
    identities = np.asarray(statements)
    return np.lexsort((identities, -np.asarray(scores)))[: min(k, len(scores))]


def fractional_top_k_weights(scores: np.ndarray, k: int) -> np.ndarray:
    """Tie-aware Top-k membership weights that sum to k.

    A tie group wholly above the boundary receives weight 1. If a tie group
    straddles the boundary, every member receives the same fractional weight.
    """
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.zeros(scores.size, dtype=np.float64)
    remaining = min(k, scores.size)
    for value in np.unique(scores)[::-1]:
        members = np.flatnonzero(scores == value)
        if remaining <= 0:
            break
        if members.size <= remaining:
            weights[members] = 1.0
            remaining -= members.size
        else:
            weights[members] = remaining / members.size
            remaining = 0
    return weights


def weighted_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.maximum(left, right).sum()
    return float(np.minimum(left, right).sum() / union) if union else 1.0


HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def deleted_source_locations(patch_path: Path) -> set[str]:
    """Return deleted Java source locations as ``qualified.Class#line``.

    This conservative pilot ground truth excludes pure-addition fixes. The full
    experiment will consume the benchmark authors' validated faulty-line map.
    """
    locations: set[str] = set()
    current_class: str | None = None
    old_line: int | None = None
    for raw in patch_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.startswith("--- a/"):
            source = raw[6:].split("\t", 1)[0].replace("\\", "/")
            relative = None
            for marker in ("src/main/java/", "src/java/", "source/", "src/"):
                if source.startswith(marker):
                    relative = source[len(marker):]
                    break
                embedded = "/" + marker
                if embedded in source:
                    relative = source.split(embedded, 1)[1]
                    break
            if relative is None and source.endswith(".java"):
                relative = source
            current_class = (
                relative[:-5].replace("/", ".")
                if relative is not None and relative.endswith(".java") else None
            )
            continue
        match = HUNK_RE.match(raw)
        if match:
            old_line = int(match.group(1))
            continue
        if old_line is None:
            continue
        if raw.startswith("-") and not raw.startswith("---"):
            if current_class is not None and raw[1:].strip():
                locations.add(f"{current_class}#{old_line}")
            old_line += 1
        elif raw.startswith("+") and not raw.startswith("+++"):
            continue
        else:
            old_line += 1
    return locations


def statement_matches_location(statement: str, location: str) -> bool:
    """Match GZoltar nested-class identities to a source-level class and line."""
    statement_class, statement_line = statement.rsplit("#", 1)
    location_class, location_line = location.rsplit("#", 1)
    return statement_line == location_line and (
        statement_class == location_class or statement_class.startswith(location_class + "$")
    )
