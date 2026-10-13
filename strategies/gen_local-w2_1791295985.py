from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    coherence: float = 0.55
    minimum_block_return: float = 0.002
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = "gen_local-w2_1791295985"

    @classmethod
    def params_type(cls) -> type[WaveParams]:
        return WaveParams

    def warmup_bars(self, params: WaveParams) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: WaveParams) -> pd.DataFrame:
        returns = data["close"].pct_change().replace([np.inf, -np.inf], np.nan)
        previous = returns.shift(4)
        dot = (returns * previous).rolling(4, min_periods=4).sum()
        recent_energy = returns.pow(2).rolling(4, min_periods=4).sum()
        previous_energy = previous.pow(2).rolling(4, min_periods=4).sum()
        denominator = np.sqrt(recent_energy * previous_energy).replace(0, np.nan)

        result = pd.DataFrame(index=data.index)
        result["coherence"] = (dot / denominator).fillna(0.0)
        result["block_return"] = returns.rolling(4, min_periods=4).sum().fillna(0.0)
        result["up_days"] = returns.gt(0).rolling(4, min_periods=4).sum().fillna(0.0)
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveParams,
    ) -> SignalFrame:
        eligible = (
            (indicators["coherence"] >= params.coherence)
            & (indicators["block_return"].abs() >= params.minimum_block_return)
            & indicators["up_days"].between(1, 3)
        )
        candidates = np.sign(indicators["block_return"].where(eligible, 0.0)).astype(int).to_numpy()
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1

        for bar in range(len(data)):
            if position:
                if bar - entry_bar >= params.hold_bars:
                    position = 0
            else:
                position = int(candidates[bar])
                if position:
                    entry_bar = bar
            decisions[bar] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
