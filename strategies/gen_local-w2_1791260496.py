from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapDepthParams:
    min_drawdown: float = 0.01
    gap_depth_fraction: float = 0.25
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GapDepthParams]):
    strategy_id = "gen_local-w2_1791260496"

    @classmethod
    def params_type(cls) -> type[GapDepthParams]:
        return GapDepthParams

    def warmup_bars(self, params: GapDepthParams) -> int:
        return 7

    def indicators(self, data: pd.DataFrame, params: GapDepthParams) -> pd.DataFrame:
        previous_close = data["close"].shift(1)
        previous_peak = data["close"].shift(1).rolling(6, min_periods=6).max()
        valid_peak = previous_peak.where(previous_peak > 0)
        depth = ((previous_peak - previous_close) / valid_peak).clip(lower=0)
        gap_depth = (data["open"] - previous_close) / valid_peak
        close_recovery = (data["close"] - previous_close) / valid_peak
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = previous_close
        out["previous_peak"] = previous_peak
        out["drawdown_depth"] = depth
        out["gap_depth_fraction"] = gap_depth / depth.where(depth > 0)
        out["close_recovery"] = close_recovery
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapDepthParams,
    ) -> SignalFrame:
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5")
        if not np.isfinite(params.min_drawdown) or params.min_drawdown <= 0:
            raise ValueError("min_drawdown must be finite and positive")
        if not np.isfinite(params.gap_depth_fraction) or params.gap_depth_fraction <= 0:
            raise ValueError("gap_depth_fraction must be finite and positive")

        eligible = indicators["drawdown_depth"].ge(params.min_drawdown)
        gap = indicators["gap_depth_fraction"]
        recovery = indicators["close_recovery"]
        long_entry = eligible & gap.le(-params.gap_depth_fraction) & recovery.gt(0)
        short_entry = eligible & gap.ge(params.gap_depth_fraction) & recovery.lt(0)
        candidates = pd.Series(0, index=data.index, dtype=int)
        candidates.loc[long_entry.fillna(False)] = 1
        candidates.loc[short_entry.fillna(False)] = -1

        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for bar, candidate in enumerate(candidates.to_numpy()):
            if position != 0 and bar == exit_bar:
                position = 0
            elif position == 0 and candidate != 0:
                position = int(candidate)
                exit_bar = bar + params.hold_bars
            signals[bar] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
