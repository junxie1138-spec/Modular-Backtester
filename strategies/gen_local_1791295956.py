from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    gap_window: int = 30
    atr_window: int = 14
    shock_sigma: float = 1.5
    quiet_ratio: float = 0.75
    trail_atr: float = 2.0
    max_hold: int = 10
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791295956"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.gap_window + 4, params.atr_window + 3)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        prev_close = data["close"].shift(1)
        gap = data["open"].div(prev_close).sub(1)
        gap_std = gap.rolling(params.gap_window).std().shift(1)
        tr = pd.concat([
            data["high"].sub(data["low"]),
            data["high"].sub(prev_close).abs(),
            data["low"].sub(prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()
        aftershock_range = data["high"].sub(data["low"]).rolling(2).mean()
        shock = ((gap_std > 0) & (gap <= -params.shock_sigma * gap_std)).shift(2).fillna(False)
        quiet = aftershock_range < params.quiet_ratio * atr.shift(3)
        return pd.DataFrame({
            "gap": gap.fillna(0.0),
            "gap_std": gap_std.fillna(0.0),
            "true_range": tr.fillna(0.0),
            "atr": atr.fillna(0.0),
            "aftershock_range": aftershock_range.fillna(0.0),
            "shock_two_bars_ago": shock.astype(bool),
            "quiet_aftershock": quiet.fillna(False),
            "entry": shock & quiet.fillna(False),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        in_position = False
        held = 0
        high_water = -np.inf
        stop = -np.inf

        for i, (close, atr, entry) in enumerate(zip(
            data["close"].to_numpy(),
            indicators["atr"].to_numpy(),
            indicators["entry"].to_numpy(),
        )):
            if in_position:
                held += 1
                high_water = max(high_water, close)
                if np.isfinite(atr) and atr > 0:
                    stop = max(stop, high_water - params.trail_atr * atr)
                if close < stop or held >= params.max_hold:
                    in_position = False
                    continue
                signal[i] = 1
            elif entry and np.isfinite(close):
                in_position = True
                held = 0
                high_water = -np.inf
                stop = -np.inf
                signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
