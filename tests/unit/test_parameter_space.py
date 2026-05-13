from __future__ import annotations

from backtester.optimize.parameter_space import expand_grid


def test_expand_grid_cartesian():
    space = {"fast": [10, 20], "slow": [50, 100]}
    combos = list(expand_grid(space))
    assert len(combos) == 4
    assert {"fast": 10, "slow": 50} in combos
    assert {"fast": 20, "slow": 100} in combos


def test_expand_grid_single_value():
    combos = list(expand_grid({"x": [1, 2, 3]}))
    assert combos == [{"x": 1}, {"x": 2}, {"x": 3}]


def test_expand_grid_empty_yields_one_empty_dict():
    assert list(expand_grid({})) == [{}]


def test_expand_grid_preserves_key_order():
    space = {"a": [1], "b": [2], "c": [3]}
    combo = list(expand_grid(space))[0]
    assert list(combo.keys()) == ["a", "b", "c"]
