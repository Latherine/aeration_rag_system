"""Retrieval metrics and a paired bootstrap, computed from one number per question:
the rank of the first correct chunk (None if it is not in the retrieved list).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def hit_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Fraction of questions whose first correct chunk is in the top k."""
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def reciprocal_ranks(ranks: Sequence[int | None]) -> np.ndarray:
    return np.array([1.0 / r if r else 0.0 for r in ranks])


def mrr(ranks: Sequence[int | None]) -> float:
    """Mean reciprocal rank: rank 1 scores 1, rank 2 scores 0.5, a miss scores 0."""
    return float(reciprocal_ranks(ranks).mean())


def paired_bootstrap(
    ranks_a: Sequence[int | None],
    ranks_b: Sequence[int | None],
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """95% interval for MRR(a) - MRR(b), resampling questions with replacement.

    Paired: each resample draws the same questions for both methods, so the
    interval reflects how the methods differ on the same questions rather than
    which questions happened to be drawn.
    """
    diffs_per_question = reciprocal_ranks(ranks_a) - reciprocal_ranks(ranks_b)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(diffs_per_question), size=(n_resamples, len(diffs_per_question)))
    means = diffs_per_question[picks].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"diff": float(diffs_per_question.mean()), "low": float(low), "high": float(high)}
