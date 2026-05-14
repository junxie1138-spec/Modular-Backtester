from collections import Counter

import pytest


def test_lhs_index_position_sampling_balanced():
    """For a 4-element list and random_n=8, each index appears 2x (+/- 1)."""
    from backtester.optimize.lhs_sampler import sample_param_space

    space = {"a": [10, 20, 30, 40]}
    samples = sample_param_space(space=space, random_n=8, seed=0)
    assert len(samples) == 8
    counts = Counter(s["a"] for s in samples)
    for v in [10, 20, 30, 40]:
        assert 1 <= counts[v] <= 3  # balanced +/- rounding


def test_lhs_seed_determinism():
    from backtester.optimize.lhs_sampler import sample_param_space

    space = {"a": [1, 2, 3], "b": [10, 20, 30, 40]}
    s1 = sample_param_space(space=space, random_n=10, seed=42)
    s2 = sample_param_space(space=space, random_n=10, seed=42)
    assert s1 == s2


def test_lhs_rejects_random_n_larger_than_cartesian_product():
    from backtester.optimize.lhs_sampler import sample_param_space

    space = {"a": [1, 2, 3], "b": [10, 20]}  # cartesian = 6
    with pytest.raises(ValueError, match="random_n"):
        sample_param_space(space=space, random_n=10, seed=0)
