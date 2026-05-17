import json
from pathlib import Path
from unittest import mock

import pytest

from factory.sortino_migration import _initial_state, _needs_rerun, _read_bundle_sortino


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


def _make_bundle(tmp_path: Path, name: str, sortino: float | None) -> Path:
    """Create a fake WFO bundle directory with a summary.json. When `sortino`
    is None the summary.json's oos_summary omits the sortino key.
    """
    bundle = tmp_path / "runs" / name
    bundle.mkdir(parents=True, exist_ok=True)
    oos = {"sharpe": 0.5, "total_return": 0.1, "max_drawdown": -0.05, "n_trades": 12}
    if sortino is not None:
        oos["sortino"] = sortino
    (bundle / "summary.json").write_text(
        json.dumps({"oos_summary": oos, "n_windows": 6}), encoding="utf-8",
    )
    return bundle


def test_read_bundle_sortino_reads_value(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, "b_read", sortino=0.77)
    assert _read_bundle_sortino(bundle) == 0.77


def test_read_bundle_sortino_missing_dir_returns_none(tmp_path: Path) -> None:
    assert _read_bundle_sortino(tmp_path / "does_not_exist") is None


def test_read_bundle_sortino_summary_without_sortino_returns_none(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, "b_nosortino", sortino=None)
    assert _read_bundle_sortino(bundle) is None


def test_read_bundle_sortino_non_dict_oos_summary_returns_none(tmp_path: Path) -> None:
    bundle = tmp_path / "runs" / "b_bad"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "summary.json").write_text(
        json.dumps({"oos_summary": None, "n_windows": 6}), encoding="utf-8",
    )
    assert _read_bundle_sortino(bundle) is None
