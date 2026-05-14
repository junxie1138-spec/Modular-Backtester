from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(slots=True, frozen=True)
class Paths:
    backtester_root: Path
    strategies_dir: Path
    configs_dir: Path
    registry_file: Path
    output_runs_dir: Path
    dedup_log: Path
    results_store: Path
    factory_log: Path
    tmp_dir: Path


@dataclass(slots=True, frozen=True)
class GenerationCfg:
    claude_cmd: str
    claude_flags: tuple[str, ...]
    generation_timeout_sec: int


@dataclass(slots=True, frozen=True)
class StagesCfg:
    stage_timeout_sec: int


@dataclass(slots=True, frozen=True)
class AlertsCfg:
    alert_threshold_metric: str
    alert_threshold: float
    telegram_bot_token: str
    telegram_chat_id: str
    dashboard_base_url: str


@dataclass(slots=True, frozen=True)
class LoopCfg:
    mode: str
    inter_cycle_sleep_sec: int
    max_cycles: int


@dataclass(slots=True, frozen=True)
class DashboardCfg:
    host: str
    port: int
    auto_refresh_sec: int


@dataclass(slots=True, frozen=True)
class Settings:
    paths: Paths
    generation: GenerationCfg
    stages: StagesCfg
    alerts: AlertsCfg
    loop: LoopCfg
    dashboard: DashboardCfg


def load_settings(path: Path) -> Settings:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    p = raw["paths"]
    root = Path(p["backtester_root"]).resolve()

    def _under_root(rel: str) -> Path:
        return (root / rel).resolve()

    paths = Paths(
        backtester_root=root,
        strategies_dir=_under_root(p["strategies_dir"]),
        configs_dir=_under_root(p["configs_dir"]),
        registry_file=_under_root(p["registry_file"]),
        output_runs_dir=_under_root(p["output_runs_dir"]),
        dedup_log=_under_root(p["dedup_log"]),
        results_store=_under_root(p["results_store"]),
        factory_log=_under_root(p["factory_log"]),
        tmp_dir=_under_root(p["tmp_dir"]),
    )
    g = raw["generation"]
    s = raw["stages"]
    a = raw["alerts"]
    lp = raw["loop"]
    d = raw["dashboard"]
    return Settings(
        paths=paths,
        generation=GenerationCfg(
            claude_cmd=g["claude_cmd"],
            claude_flags=tuple(g["claude_flags"]),
            generation_timeout_sec=int(g["generation_timeout_sec"]),
        ),
        stages=StagesCfg(stage_timeout_sec=int(s["stage_timeout_sec"])),
        alerts=AlertsCfg(
            alert_threshold_metric=a["alert_threshold_metric"],
            alert_threshold=float(a["alert_threshold"]),
            telegram_bot_token=a["telegram_bot_token"],
            telegram_chat_id=a["telegram_chat_id"],
            dashboard_base_url=a["dashboard_base_url"],
        ),
        loop=LoopCfg(
            mode=lp["mode"],
            inter_cycle_sleep_sec=int(lp["inter_cycle_sleep_sec"]),
            max_cycles=int(lp["max_cycles"]),
        ),
        dashboard=DashboardCfg(
            host=d["host"], port=int(d["port"]),
            auto_refresh_sec=int(d["auto_refresh_sec"]),
        ),
    )
