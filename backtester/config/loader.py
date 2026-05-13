from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from backtester.config.models import (
    DataConfig,
    ExecutionConfig,
    OptimizationConfig,
    PortfolioConfig,
    RunConfig,
    WFOConfig,
)
from backtester.core.exceptions import ConfigError

PathLike = Union[str, Path]


def _require(d: Dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise ConfigError(f"missing required field {key!r} in {where}")
    return d[key]


def _from_dict(raw: Dict[str, Any]) -> RunConfig:
    try:
        data_raw = _require(raw, "data", "config root")
        data = DataConfig(
            symbols=list(_require(data_raw, "symbols", "data")),
            timeframe=_require(data_raw, "timeframe", "data"),
            start=_require(data_raw, "start", "data"),
            end=_require(data_raw, "end", "data"),
            source=data_raw.get("source", "csv"),
            root=data_raw.get("root", "data/raw"),
        )
        execution = ExecutionConfig(**(raw.get("execution") or {}))
        portfolio = PortfolioConfig(**(raw.get("portfolio") or {}))

        opt = None
        if raw.get("optimization"):
            opt_raw = raw["optimization"]
            opt = OptimizationConfig(
                objective=opt_raw.get("objective", "sharpe"),
                param_space=dict(opt_raw.get("param_space", {})),
            )

        wfo = None
        if raw.get("wfo"):
            wfo = WFOConfig(**raw["wfo"])

        return RunConfig(
            run_name=_require(raw, "run_name", "config root"),
            strategy=_require(raw, "strategy", "config root"),
            strategy_params=dict(raw.get("strategy_params", {})),
            data=data,
            execution=execution,
            portfolio=portfolio,
            optimization=opt,
            wfo=wfo,
            output_root=raw.get("output_root", "output/runs"),
            seed=int(raw.get("seed", 0)),
        )
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"failed to parse config: {exc}") from exc


def load_run_config(path: PathLike) -> RunConfig:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config not found: {p}")
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")
    return _from_dict(raw)


def dump_run_config(rc: RunConfig, path: PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(rc)
    if payload.get("optimization") is None:
        payload.pop("optimization", None)
    if payload.get("wfo") is None:
        payload.pop("wfo", None)
    p.write_text(yaml.safe_dump(payload, sort_keys=False))
