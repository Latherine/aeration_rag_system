import pytest

from aeration_qa.metrics import hit_at_k, mrr, paired_bootstrap


def test_hit_at_k_counts_ranks_within_k():
    ranks = [1, 3, 4, None]
    assert hit_at_k(ranks, 1) == 0.25
    assert hit_at_k(ranks, 3) == 0.5
    assert hit_at_k(ranks, 10) == 0.75


def test_mrr_scores_misses_as_zero():
    assert mrr([1, 2, None, 4]) == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)


def test_identical_methods_have_zero_difference_and_zero_width():
    result = paired_bootstrap([1, 2, None], [1, 2, None])
    assert result == {"diff": 0.0, "low": 0.0, "high": 0.0}


def test_consistently_better_method_has_interval_above_zero():
    better = [1] * 20
    worse = [5] * 20
    result = paired_bootstrap(better, worse)
    assert result["diff"] == pytest.approx(0.8)
    assert result["low"] > 0


def test_bootstrap_is_reproducible():
    a, b = [1, None, 2, 3, None, 1], [2, 1, None, 3, 4, None]
    assert paired_bootstrap(a, b, seed=7) == paired_bootstrap(a, b, seed=7)
