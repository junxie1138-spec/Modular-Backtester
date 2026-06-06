from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

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
        "oos_sortino": float(oos["sortino"]),
        "oos_total_return": float(oos["total_return"]),
        "oos_max_drawdown": float(oos["max_drawdown"]),
        "oos_n_trades": int(oos["n_trades"]),
        "parameter_stability": dict(raw.get("parameter_stability", {})),
        "n_windows": int(raw["n_windows"]),
        "run_bundle_path": bundle_path.as_posix(),
    }


# ---------------------------------------------------------------------------
# Subprocess stage wrappers (Task 9 / §5.7, R2)
# ---------------------------------------------------------------------------

STAGE_SUFFIX: dict[str, str] = {"backtest": "", "optimize": "_grid", "wfo": "_wfo"}
STAGE_MODULE: dict[str, str] = {
    "backtest": "backtester.runners.run_backtest",
    "optimize": "backtester.runners.run_optimize",
    "wfo": "backtester.runners.run_wfo",
}


@dataclass(slots=True, frozen=True)
class StageResult:
    stage: str
    parsed: dict[str, Any]
    bundle_path: Path
    raw_summary: dict[str, Any]


def build_stage_config(
    *, canonical_path: Path, strategy_id: str, stage: str, tmp_dir: Path,
) -> Path:
    """Clone the canonical YAML to <tmp_dir>/<strategy_id>/<stage>.yaml with
    `run_name` rewritten so each stage's output bundle is distinct (R2).
    """
    if stage not in STAGE_SUFFIX:
        raise ValueError(f"unknown stage: {stage}")
    raw = yaml.safe_load(canonical_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise StageError(f"canonical config is not a mapping: {canonical_path}")
    suffix = STAGE_SUFFIX[stage]
    raw["run_name"] = f"{strategy_id}{suffix}"
    out_dir = tmp_dir / strategy_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stage}.yaml"
    out_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return out_path


def _run_stage(
    *,
    stage: str,
    canonical_config: Path,
    strategy_id: str,
    output_runs_dir: Path,
    tmp_dir: Path,
    timeout_sec: int,
    backtester_root: Path | None = None,
) -> StageResult:
    """Internal: run one stage subprocess and parse its summary.json."""
    stage_cfg = build_stage_config(
        canonical_path=canonical_config,
        strategy_id=strategy_id,
        stage=stage,
        tmp_dir=tmp_dir,
    )
    cmd = [sys.executable, "-m", STAGE_MODULE[stage], "--config", str(stage_cfg)]
    log.info("running stage=%s cmd=%s", stage, cmd)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
            cwd=str(backtester_root) if backtester_root else None,
            encoding="utf-8",
        )
    except subprocess.TimeoutExpired as exc:
        raise StageError(f"stage={stage} timed out after {timeout_sec}s") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-2000:]
        raise StageError(f"stage={stage} exit={proc.returncode}; stderr tail:\n{tail}")

    run_name = f"{strategy_id}{STAGE_SUFFIX[stage]}"
    try:
        bundle = find_latest_bundle(output_runs_dir=output_runs_dir, run_name=run_name)
    except BundleNotFound as exc:
        raise StageError(f"stage={stage}: {exc}") from exc

    summary_path = bundle / "summary.json"
    if not summary_path.exists():
        raise StageError(f"stage={stage}: summary.json missing in {bundle}")
    try:
        raw_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StageError(f"stage={stage}: summary.json not valid JSON: {exc}") from exc

    try:
        if stage == "backtest":
            parsed = parse_backtest_summary(raw_summary, bundle_path=bundle)
        elif stage == "optimize":
            parsed = parse_optimize_summary(raw_summary, bundle_path=bundle)
        elif stage == "wfo":
            parsed = parse_wfo_summary(raw_summary, bundle_path=bundle)
        else:  # pragma: no cover — guarded above
            raise StageError(f"unknown stage: {stage}")
    except KeyError as exc:
        raise StageError(f"stage={stage}: summary.json missing expected key: {exc}") from exc

    return StageResult(stage=stage, parsed=parsed, bundle_path=bundle, raw_summary=raw_summary)


def run_backtest_stage(**kwargs) -> StageResult:
    return _run_stage(stage="backtest", **kwargs)


def run_optimize_stage(**kwargs) -> StageResult:
    return _run_stage(stage="optimize", **kwargs)


def run_wfo_stage(**kwargs) -> StageResult:
    return _run_stage(stage="wfo", **kwargs)
