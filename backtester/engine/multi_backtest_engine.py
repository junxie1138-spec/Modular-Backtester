from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from backtester.engine.multi_portfolio import (
    MultiSymbolPortfolioSimulator, MultiSymbolResult,
)


def _align_panel(
    data: dict[str, pd.DataFrame],
    aux_data: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Reindex all panels to the union of their datetime indices.

    Missing bars (e.g., pre-IPO for late symbols) are padded with NaN. The
    strategy and simulator are warmup-aware and skip NaN bars naturally.
    """
    all_indices: set[pd.Timestamp] = set()
    for df in data.values():
        all_indices.update(df.index)
    for df in aux_data.values():
        all_indices.update(df.index)
    if not all_indices:
        return data, aux_data
    union_index = pd.DatetimeIndex(sorted(all_indices))
    aligned_data = {sym: df.reindex(union_index) for sym, df in data.items()}
    aligned_aux = {sym: df.reindex(union_index) for sym, df in aux_data.items()}
    return aligned_data, aligned_aux


@dataclass
class MultiSymbolBacktestEngine:
    simulator: MultiSymbolPortfolioSimulator

    def run(
        self,
        *,
        strategy: Any,
        symbols: list[str],
        data: dict[str, pd.DataFrame],
        sectors: dict[str, str],
        aux_data: dict[str, pd.DataFrame],
        params: Any,
        regime_config: Optional[Any] = None,
    ) -> MultiSymbolResult:
        # Align all symbol panels (and aux) to the union of their indices.
        # IPO-late symbols (e.g., COIN, PLTR) get NaN-padded for pre-IPO dates.
        # The simulator's per-bar loop iterates against this union; indicators
        # and the strategy already handle NaN-bars by emitting 0 (warmup-aware).
        data, aux_data = _align_panel(data, aux_data)

        # Pre-compute indicators for ALL strategies (per-bar AND non-per-bar).
        indicators_panel: dict[str, Any] = {}
        if hasattr(strategy, "indicators"):
            for sym in symbols:
                indicators_panel[sym] = strategy.indicators(data[sym], params)

        # Pre-compute per-symbol signals for non-per-bar strategies.
        signals: dict[str, pd.DataFrame] = {}
        if not getattr(strategy, "uses_per_bar", False):
            for sym in symbols:
                signals[sym] = strategy.generate_signals_for_symbol(
                    data=data[sym], indicators=indicators_panel[sym], params=params,
                )
        else:
            # Per-bar strategies: provide an empty signals frame to be overwritten.
            for sym in symbols:
                idx = data[sym].index
                signals[sym] = pd.DataFrame(
                    {"signal": [0.0] * len(idx), "size": [1.0] * len(idx)}, index=idx,
                )

        return self.simulator.simulate(
            symbols=symbols, data=data, sectors=sectors, signals=signals,
            aux_data=aux_data, regime_config=regime_config,
            strategy=strategy if getattr(strategy, "uses_per_bar", False) else None,
            strategy_params=params,
            indicators_panel=indicators_panel,
        )
