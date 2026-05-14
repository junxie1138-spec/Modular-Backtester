from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from backtester.engine.multi_portfolio import (
    MultiSymbolPortfolioSimulator, MultiSymbolResult,
)


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
