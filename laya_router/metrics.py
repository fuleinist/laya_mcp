"""Paired metrics: accuracy with an interval, per-class recall, exact McNemar, and ECE.

Accuracy alone cannot settle a router comparison. Two backends that agree on most items and
disagree on a handful will land within noise of each other, and the interesting quantity is the
*paired* one: how many items each got right that the other got wrong. Upstream, a 0.600-vs-0.600
tie was reported as 84 disagreements split 38–38 with McNemar p = 1.00 — that is the reporting
shape this module exists to produce, without scipy.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Sequence


def wilson(successes: int, total: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% at z=1.96)."""
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def binom_pmf(n: int, k: int) -> float:
    return math.comb(n, k) * 0.5 ** n


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p over the discordant pairs (b = first-only-right, c = second-only).

    Exact (binomial) rather than chi-square: the discordant count in an eval this size is small
    enough that the asymptotic test is not trustworthy.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(binom_pmf(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail)


def ece(probs: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float | None:
    """Expected calibration error over equal-width bins; None when nothing was scored."""
    pairs = [(p, c) for p, c in zip(probs, correct) if p is not None]
    if not pairs:
        return None
    total = len(pairs)
    error = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [(p, c) for p, c in pairs if (lo < p <= hi) or (b == 0 and p == 0.0)]
        if not bucket:
            continue
        mean_conf = sum(p for p, _ in bucket) / len(bucket)
        mean_acc = sum(1 for _, c in bucket if c) / len(bucket)
        error += (len(bucket) / total) * abs(mean_acc - mean_conf)
    return error


def percentile(values: Iterable[float], q: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return vals[int(pos)]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def confusion(gold: Sequence[Any], pred: Sequence[Any], labels: Sequence[str]) -> dict[str, dict[str, int]]:
    matrix = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold, pred):
        if g in matrix and p in matrix[g]:
            matrix[g][p] += 1
    return matrix


def per_class_recall(gold: Sequence[Any], pred: Sequence[Any], labels: Sequence[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for label in labels:
        idx = [i for i, g in enumerate(gold) if g == label]
        if not idx:
            out[label] = None
            continue
        out[label] = sum(1 for i in idx if pred[i] == label) / len(idx)
    return out


def summarise_single(gold: Sequence[Any], pred: Sequence[Any], labels: Sequence[str],
                     probs: Sequence[float | None] | None = None,
                     latencies: Sequence[float | None] | None = None) -> dict[str, Any]:
    """One backend's score on one question, with an interval and a class breakdown."""
    total = len(gold)
    correct = [i for i, (g, p) in enumerate(zip(gold, pred)) if g == p and g is not None]
    hits = len(correct)
    lo, hi = wilson(hits, total)
    out: dict[str, Any] = {
        "n": total,
        "correct": hits,
        "accuracy": (hits / total) if total else None,
        "accuracy_ci95": [lo, hi],
        "per_class_recall": per_class_recall(gold, pred, labels),
        "confusion": confusion(gold, pred, labels),
        "unanswered": sum(1 for p in pred if p is None),
    }
    if probs is not None:
        flags = [i in correct for i in range(total)]
        out["ece"] = ece(list(probs), flags)
    if latencies is not None:
        out["latency_ms"] = {"p50": percentile(latencies, 0.5), "p99": percentile(latencies, 0.99),
                             "mean": (sum(v for v in latencies if v is not None) /
                                      max(1, sum(1 for v in latencies if v is not None)))}
    return out


def paired(a_gold: Sequence[Any], a_pred: Sequence[Any], b_pred: Sequence[Any],
           labels: Sequence[str]) -> dict[str, Any]:
    """Paired comparison of two backends on the same items, with exact McNemar.

    `a_only` / `b_only` count the discordant items: the questions where one backend was right and
    the other was wrong. Those are the only items that carry information about the difference.
    """
    total = len(a_gold)
    both = a_only = b_only = neither = 0
    for g, pa, pb in zip(a_gold, a_pred, b_pred):
        ca, cb = (pa == g and g is not None), (pb == g and g is not None)
        if ca and cb:
            both += 1
        elif ca:
            a_only += 1
        elif cb:
            b_only += 1
        else:
            neither += 1
    return {
        "n": total,
        "both_correct": both,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "neither_correct": neither,
        "disagreements_on_outcome": a_only + b_only,
        "mcnemar_exact_p": mcnemar_exact(a_only, b_only),
        "delta_accuracy": ((a_only - b_only) / total) if total else None,
        "label_counts": dict(Counter(labels)),
    }