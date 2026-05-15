from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path


# node_id is used in filenames and git-safe paths, so it is constrained.
_NODE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


@dataclass(slots=True, frozen=True)
class Paths:
    backtester_root: Path
    strategies_dir: Path
    configs_dir: Path
    registry_file: Path
    output_runs_dir: Path
    dedup_dir: Path
    results_dir: Path
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
class PromotionCfg:
    enabled: bool
    tickers: tuple[str, ...]
    data_source: str
    min_avg_sharpe: float
    trigger_metric: str
    trigger_threshold: float


@dataclass(slots=True, frozen=True)
class ScreeningCfg:
    enabled: bool
    min_optimize_score: float


@dataclass(slots=True, frozen=True)
class SyncCfg:
    enabled: bool
    branch: str
    remote: str
    push_retries: int


@dataclass(slots=True, frozen=True)
class Settings:
    node_id: str
    paths: Paths
    generation: GenerationCfg
    stages: StagesCfg
    alerts: AlertsCfg
    loop: LoopCfg
    dashboard: DashboardCfg
    promotion: PromotionCfg
    screening: ScreeningCfg
    sync: SyncCfg


def load_settings(path: Path) -> Settings:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    # Merge an optional sibling settings.local.toml over the base. The local
    # file is gitignored and holds secrets (Telegram token, API keys) so they
    # never land in version control. Merge is shallow per top-level section.
    local_path = path.parent / "settings.local.toml"
    if local_path.exists():
        local = tomllib.loads(local_path.read_text(encoding="utf-8"))
        for section, overrides in local.items():
            if isinstance(overrides, dict) and isinstance(raw.get(section), dict):
                raw[section] = {**raw[section], **overrides}
            else:
                raw[section] = overrides
    node_id = str(raw.get("node_id", "local"))
    if not _NODE_ID_RE.match(node_id):
        raise ValueError(
            f"invalid node_id {node_id!r}: must match ^[a-z0-9][a-z0-9-]*$ "
            f"(lowercase letters, digits and hyphens; not starting with a hyphen). "
            f"Set it in factory/config/settings.local.toml."
        )
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
        dedup_dir=_under_root(p["dedup_dir"]),
        results_dir=_under_root(p["results_dir"]),
        factory_log=_under_root(p["factory_log"]),
        tmp_dir=_under_root(p["tmp_dir"]),
    )
    g = raw["generation"]
    s = raw["stages"]
    a = raw["alerts"]
    lp = raw["loop"]
    d = raw["dashboard"]
    pr = raw.get("promotion", {}) or {}
    sc = raw.get("screening", {}) or {}
    sy = raw.get("sync", {}) or {}
    return Settings(
        node_id=node_id,
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
        promotion=PromotionCfg(
            enabled=bool(pr.get("enabled", False)),
            tickers=tuple(pr.get("tickers", ())),
            data_source=str(pr.get("data_source", "yfinance")),
            min_avg_sharpe=float(pr.get("min_avg_sharpe", 0.7)),
            trigger_metric=str(pr.get("trigger_metric", "wfo.oos_sharpe")),
            trigger_threshold=float(pr.get("trigger_threshold", 1.0)),
        ),
        screening=ScreeningCfg(
            enabled=bool(sc.get("enabled", False)),
            min_optimize_score=float(sc.get("min_optimize_score", 1.3)),
        ),
        sync=SyncCfg(
            enabled=bool(sy.get("enabled", False)),
            branch=str(sy.get("branch", "factory-pool")),
            remote=str(sy.get("remote", "origin")),
            push_retries=int(sy.get("push_retries", 5)),
        ),
    )
