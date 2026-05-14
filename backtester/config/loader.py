from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from backtester.config.models import (
    CircuitBreakerConfig,
    DataConfig,
    ExecutionConfig,
    OptimizationConfig,
    PortfolioConfig,
    RegimesConfig,
    RunConfig,
    SpyEmaRegimeConfig,
    VixRegimeConfig,
    WFOConfig,
)
from backtester.core.exceptions import ConfigError

PathLike = Union[str, Path]


def _require(d: Dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise ConfigError(f"missing required field {key!r} in {where}")
    return d[key]


def _parse_regimes(regimes_raw: Dict[str, Any]) -> RegimesConfig:
    """Parse the `regimes:` block into a RegimesConfig dataclass."""
    spy_raw = regimes_raw.get("spy_ema", {}) or {}
    vix_raw = regimes_raw.get("vix", {}) or {}
    cb_raw = regimes_raw.get("circuit_breaker", {}) or {}
    return RegimesConfig(
        spy_ema=SpyEmaRegimeConfig(**spy_raw),
        vix=VixRegimeConfig(**vix_raw),
        circuit_breaker=CircuitBreakerConfig(**cb_raw),
    )


def _resolve_universe_path(raw_path: str, config_file_path: Path) -> str:
    """Resolve universe_path relative to the config file's directory."""
    p = Path(raw_path)
    if p.is_absolute():
        return str(p)
    # Relative paths are resolved relative to the config file's directory.
    resolved = (config_file_path.parent / p).resolve()
    return str(resolved)


def _from_dict(raw: Dict[str, Any], config_file_path: Optional[Path] = None) -> RunConfig:
    try:
        data_raw = _require(raw, "data", "config root")
        # `symbols` is optional when universe_path is provided (v0.4.0 multi-symbol runs).
        has_universe = bool(raw.get("universe_path"))
        symbols_raw = data_raw.get("symbols", [])
        if not has_universe and not symbols_raw:
            _require(data_raw, "symbols", "data")  # raises ConfigError with clear message
        data = DataConfig(
            symbols=list(symbols_raw) if symbols_raw else [],
            timeframe=_require(data_raw, "timeframe", "data"),
            start=_require(data_raw, "start", "data"),
            end=_require(data_raw, "end", "data"),
            source=data_raw.get("source", "csv"),
            root=data_raw.get("root", "data/raw"),
            auto_adjust=bool(data_raw.get("auto_adjust", True)),
            aux_symbols=list(data_raw.get("aux_symbols", [])),
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

        regimes = None
        if raw.get("regimes"):
            regimes = _parse_regimes(raw["regimes"])

        universe_path = None
        if raw.get("universe_path"):
            raw_up = raw["universe_path"]
            if config_file_path is not None:
                universe_path = _resolve_universe_path(raw_up, config_file_path)
            else:
                universe_path = raw_up

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
            universe_path=universe_path,
            regimes=regimes,
        )
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"failed to parse config: {exc}") from exc


def load_run_config(path: PathLike) -> RunConfig:
    p = Path(path).resolve()
    if not p.exists():
        raise ConfigError(f"config not found: {p}")
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")
    return _from_dict(raw, config_file_path=p)


def dump_run_config(rc: RunConfig, path: PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(rc)
    if payload.get("optimization") is None:
        payload.pop("optimization", None)
    if payload.get("wfo") is None:
        payload.pop("wfo", None)
    p.write_text(yaml.safe_dump(payload, sort_keys=False))
