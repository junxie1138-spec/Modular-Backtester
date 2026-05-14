from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class StageError(RuntimeError):
    """A backtester stage exited non-zero, timed out, or produced no summary.json."""


class BundleNotFound(StageError):
    """No bundle dir matching the expected run_name."""


def find_latest_bundle(*, output_runs_dir: Path, run_name: str) -> Path:
    """Return the newest directory under output_runs_dir whose name ends with
    `_<run_name>` (after the YYYYMMDD_HHMM prefix).
    """
    if not output_runs_dir.exists():
        raise BundleNotFound(f"output_runs_dir does not exist: {output_runs_dir}")
    candidates = [
        d for d in output_runs_dir.iterdir()
        if d.is_dir() and d.name.endswith(f"_{run_name}")
    ]
    if not candidates:
        raise BundleNotFound(
            f"no bundle found in {output_runs_dir} for run_name={run_name!r}"
        )
    # Pick the one with the newest mtime (handles minute-resolution timestamp ties).
    return max(candidates, key=lambda d: d.stat().st_mtime)


def parse_backtest_summary(raw: dict[str, Any], *, bundle_path: Path) -> dict[str, Any]:
    """Map backtest summary.json (flat metrics) onto the factory record shape."""
    return {
        "sharpe": float(raw["sharpe"]),
        "total_return": float(raw["total_return"]),
        "max_drawdown": float(raw["max_drawdown"]),
        "win_rate": float(raw["win_rate"]),
        "n_trades": int(raw["n_trades"]),
        "run_bundle_path": bundle_path.as_posix(),
    }


def parse_optimize_summary(raw: dict[str, Any], *, bundle_path: Path) -> dict[str, Any]:
    """Map optimize summary.json onto the factory record shape.

    Real shape: {best_params, best_score_objective, best_summary{...}}.
    best_score = best_summary[best_score_objective].
    """
    best_summary = raw["best_summary"]
    objective = raw["best_score_objective"]
    if objective not in best_summary:
        raise KeyError(
            f"objective {objective!r} not found in best_summary keys "
            f"{sorted(best_summary.keys())}"
        )
    return {
        "best_params": dict(raw["best_params"]),
        "objective": objective,
        "best_score": float(best_summary[objective]),
        "run_bundle_path": bundle_path.as_posix(),
    }


def parse_wfo_summary(raw: dict[str, Any], *, bundle_path: Path) -> dict[str, Any]:
    """Map WFO summary.json (nested oos_summary) onto the factory record shape."""
    oos = raw["oos_summary"]
    return {
        "oos_sharpe": float(oos["sharpe"]),
        "oos_total_return": float(oos["total_return"]),
        "oos_max_drawdown": float(oos["max_drawdown"]),
        "oos_n_trades": int(oos["n_trades"]),
        "parameter_stability": dict(raw.get("parameter_stability", {})),
        "n_windows": int(raw["n_windows"]),
        "run_bundle_path": bundle_path.as_posix(),
    }
