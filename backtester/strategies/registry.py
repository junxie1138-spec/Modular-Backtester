from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Dict, Type

from backtester.strategies.base import BaseStrategy

log = logging.getLogger(__name__)

STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {}


def register_strategy(cls: Type[BaseStrategy]) -> Type[BaseStrategy]:
    """Register a strategy class by its `strategy_id`."""
    if not getattr(cls, "strategy_id", None):
        raise ValueError(f"{cls.__name__} is missing a non-empty `strategy_id`")
    STRATEGY_REGISTRY[cls.strategy_id] = cls
    return cls


def get_strategy_class(strategy_id: str) -> Type[BaseStrategy]:
    if strategy_id not in STRATEGY_REGISTRY:
        raise KeyError(
            f"Strategy {strategy_id!r} is not registered. "
            f"Known: {sorted(STRATEGY_REGISTRY)}"
        )
    return STRATEGY_REGISTRY[strategy_id]


def discover_generated_strategies() -> list[str]:
    """Import every `strategies/gen_*.py` and register its GeneratedStrategy.

    The factory's distributed design (no per-strategy registry edit) relies on
    this: every machine's generated strategies are picked up automatically.

    - `importlib.invalidate_caches()` is called first so a strategy file
      written earlier in the same process is visible.
    - Only `gen_*.py` is globbed; curated strategies are never touched here.
    - Filenames are sorted before import so registration order is the same
      on every machine and every test run.
    - Each import is wrapped: one broken generated module is skipped (its
      filename and full exception are logged), never aborting the import.

    Returns the list of module names imported, in import order.
    """
    import strategies as _strategies_pkg
    pkg_dir = Path(_strategies_pkg.__file__).resolve().parent
    importlib.invalidate_caches()
    imported: list[str] = []
    for path in sorted(pkg_dir.glob("gen_*.py")):
        module_name = f"strategies.{path.stem}"
        try:
            module = importlib.import_module(module_name)
            register_strategy(module.GeneratedStrategy)
            imported.append(module_name)
        except Exception as exc:
            log.warning(
                "registry auto-discovery: skipping %s — %r",
                path.name, exc, exc_info=True,
            )
    return imported


# --- Curated strategy registrations (explicit, predictable order) ---
from strategies.sma_cross import SMACrossStrategy  # noqa: E402
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy  # noqa: E402
from strategies.breakout_20d import Breakout20DStrategy  # noqa: E402
from strategies.rsi_long_short import RSILongShortStrategy  # noqa: E402
from strategies.momentum_streak import MomentumStreakStrategy  # noqa: E402
from strategies.mean_reversion_atr import MeanReversionAtrStrategy  # noqa: E402

register_strategy(SMACrossStrategy)
register_strategy(RSIMeanReversionStrategy)
register_strategy(Breakout20DStrategy)
register_strategy(RSILongShortStrategy)
register_strategy(MomentumStreakStrategy)
register_strategy(MeanReversionAtrStrategy)

# --- Generated strategies: auto-discovered from strategies/gen_*.py ---
discover_generated_strategies()
