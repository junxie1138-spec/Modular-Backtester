import json
from pathlib import Path
from unittest import mock

import pytest

from factory.sortino_migration import _initial_state, _needs_rerun


def test_needs_rerun_same_side_is_false() -> None:
    # Both Sharpe and Sortino below the threshold -> verdict unchanged.
    assert _needs_rerun(0.4, 0.6, 1.0) is False
    # Both above -> verdict unchanged.
    assert _needs_rerun(1.5, 1.2, 1.0) is False


def test_needs_rerun_opposite_sides_is_true() -> None:
    # Sharpe below, Sortino at/above -> the metric swap flips the verdict.
    assert _needs_rerun(0.9, 1.1, 1.0) is True
    # Sharpe above, Sortino below.
    assert _needs_rerun(1.3, 0.8, 1.0) is True


def test_initial_state_below_threshold_is_na() -> None:
    assert _initial_state(
        has_promotion_block=False, oos_sortino=0.5,
        promotion_enabled=True, trigger_threshold=1.0,
    ) == "n/a"


def test_initial_state_eligible_is_pending() -> None:
    assert _initial_state(
        has_promotion_block=False, oos_sortino=1.5,
        promotion_enabled=True, trigger_threshold=1.0,
    ) == "pending"


def test_initial_state_promotion_disabled_is_na() -> None:
    assert _initial_state(
        has_promotion_block=False, oos_sortino=1.5,
        promotion_enabled=False, trigger_threshold=1.0,
    ) == "n/a"


def test_initial_state_existing_promotion_block_is_done() -> None:
    assert _initial_state(
        has_promotion_block=True, oos_sortino=1.5,
        promotion_enabled=True, trigger_threshold=1.0,
    ) == "done"
