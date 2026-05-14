from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from backtester.core.exceptions import ConfigError


# data/sector_map.csv lives 2 levels above this file:
# backtester/config/universe.py -> backtester/config/ -> backtester/ -> repo_root/
_SECTOR_MAP_PATH = Path(__file__).resolve().parents[2] / "data" / "sector_map.csv"


@dataclass(slots=True)
class ResolvedSymbolConfig:
    symbol: str
    sector: str
    effective_params: dict[str, Any] = field(default_factory=dict)


def _load_sector_map() -> dict[str, str]:
    if not _SECTOR_MAP_PATH.exists():
        return {}
    with _SECTOR_MAP_PATH.open(newline="", encoding="utf-8") as f:
        return {row["symbol"]: row["sector"] for row in csv.DictReader(f)}


def load_universe_config(
    *,
    path: Path,
    global_params: dict[str, Any],
) -> dict[str, ResolvedSymbolConfig]:
    """Parse universe.yaml and resolve per-symbol sector + overrides.

    Resolution precedence (low -> high):
      1. global_params (from run YAML's strategy_params)
      2. per-name overrides (from universe.yaml)

    Sector resolution:
      1. data/sector_map.csv lookup
      2. universe.yaml inline `sector` field (wins if present)
      3. ConfigError if neither resolves to a non-empty string

    Returns dict[symbol, ResolvedSymbolConfig].
    """
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    universe_dict = doc.get("universe", {})
    if not isinstance(universe_dict, dict):
        raise ConfigError(f"{path}: top-level `universe:` must be a mapping")

    sector_map = _load_sector_map()
    allowed_keys = set(global_params)
    out: dict[str, ResolvedSymbolConfig] = {}

    for symbol, meta in universe_dict.items():
        meta = meta or {}
        overrides = meta.get("overrides", {}) or {}
        unknown = set(overrides) - allowed_keys
        if unknown:
            raise ConfigError(
                f"universe.yaml: {symbol} overrides reference keys not in "
                f"strategy_params: {sorted(unknown)}"
            )
        inline_sector = meta.get("sector")
        sector = inline_sector if inline_sector else sector_map.get(symbol)
        if not sector:
            raise ConfigError(
                f"universe.yaml: {symbol} has no sector (not in sector_map.csv "
                f"and no inline `sector` field)"
            )
        effective = dict(global_params)
        effective.update(overrides)
        out[symbol] = ResolvedSymbolConfig(
            symbol=symbol, sector=sector, effective_params=effective,
        )
    return out
