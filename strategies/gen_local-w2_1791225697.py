from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    peak_window: int = 63
    gap_window: int = 20
    min_drawdown: float = 0.05
    gap_mult: float = 1.5
    recovery_frac: float = 0.08
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791225697"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.peak_window + 1, params.gap_window + 2)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        prev_close = data["close"].shift(1)
        prior_peak = data["close"].rolling(params.peak_window).max().shift(1)
        gap = (data["open"] / prev_close - 1.0).replace([np.inf, -np.inf], np.nan)
        typical_gap = gap.abs().rolling(params.gap_window).median().shift(1)
        out = pd.DataFrame(index=data.index)
        out["gap_score"] = (gap / typical_gap.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["prior_depth"] = (1.0 - prev_close / prior_peak).clip(lower=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["depth_repair"] = ((data["close"] - prev_close) / prior_peak).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        depth = indicators["prior_depth"]
        repair = indicators["depth_repair"]
        gap = indicators["gap_score"]
        underwater = depth >= params.min_drawdown
        long_entry = (underwater & (gap <= -params.gap_mult) & (repair >= depth * params.recovery_frac)).to_numpy()
        short_entry = (underwater & (gap >= params.gap_mult) & (repair <= -depth * params.recovery_frac)).to_numpy()

        raw = np.zeros(len(data), dtype=int)
        side, exit_index = 0, -1
        for i in range(len(data)):
            if side and i >= exit_index:
                side = 0
            elif side == 0:
                if long_entry[i]:
                    side, exit_index = 1, i + params.hold_bars
                elif short_entry[i]:
                    side, exit_index = -1, i + params.hold_bars
            raw[i] = side

        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
