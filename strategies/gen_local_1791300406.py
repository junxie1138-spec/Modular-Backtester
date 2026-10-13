from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletBreakoutParams:
    breakout_bars: int = 20
    energy_baseline: int = 60
    energy_multiple: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[WaveletBreakoutParams]):
    strategy_id = "gen_local_1791300406"

    @classmethod
    def params_type(cls) -> type[WaveletBreakoutParams]:
        return WaveletBreakoutParams

    @classmethod
    def warmup_bars(cls, params: WaveletBreakoutParams) -> int:
        return max(16 + params.energy_baseline - 1, params.breakout_bars + 1)

    def indicators(self, data: pd.DataFrame, params: WaveletBreakoutParams) -> pd.DataFrame:
        log_close = np.log(data["close"].where(data["close"] > 0))
        energy = pd.Series(0.0, index=data.index)
        for scale in (2, 4, 8):
            block_mean = log_close.rolling(scale, min_periods=scale).mean()
            detail = np.sqrt(scale / 2.0) * (block_mean - block_mean.shift(scale))
            energy = energy + detail.pow(2)

        baseline = energy.rolling(params.energy_baseline, min_periods=params.energy_baseline).median()
        out = pd.DataFrame(index=data.index)
        out["energy_ready"] = baseline.notna() & baseline.gt(0)
        out["energy"] = energy.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["energy_baseline"] = baseline.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["prior_high"] = data["high"].shift(1).rolling(params.breakout_bars, min_periods=params.breakout_bars).max().fillna(np.inf)
        out["prior_low"] = data["low"].shift(1).rolling(params.breakout_bars, min_periods=params.breakout_bars).min().fillna(-np.inf)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletBreakoutParams,
    ) -> SignalFrame:
        shock = indicators["energy_ready"] & indicators["energy"].gt(
            params.energy_multiple * indicators["energy_baseline"]
        )
        long_entry = (shock & data["close"].gt(indicators["prior_high"])).to_numpy()
        short_entry = (shock & data["close"].lt(indicators["prior_low"])).to_numpy()

        signals = []
        position = 0
        age = 0
        for go_long, go_short in zip(long_entry, short_entry):
            if position:
                age += 1
                if age >= params.hold_bars:
                    position = 0
            elif go_long or go_short:
                position = 1 if go_long else -1
                age = 0
            signals.append(position)

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(signals, index=data.index, dtype=int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
