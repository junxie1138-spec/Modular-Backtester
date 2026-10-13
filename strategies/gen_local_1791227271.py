from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapDecayParams:
    streak_length: int = 3
    gap_floor_atr: float = 0.05
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[GapDecayParams]):
    strategy_id = "gen_local_1791227271"

    @classmethod
    def params_type(cls):
        return GapDecayParams

    def warmup_bars(self, params: GapDecayParams) -> int:
        return 200

    def indicators(self, data: pd.DataFrame, params: GapDecayParams) -> pd.DataFrame:
        prev_close = data["close"].shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - prev_close).abs(),
             (data["low"] - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(14, min_periods=14).mean()
        gap = data["open"] - prev_close
        shrinking = gap.abs() < gap.shift(1).abs()
        large_enough = gap.abs() >= params.gap_floor_atr * atr.shift(1)
        down = (gap < 0) & (gap.shift(1) < 0) & shrinking & large_enough
        up = (gap > 0) & (gap.shift(1) > 0) & shrinking & large_enough

        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean().fillna(0.0)
        out["atr"] = atr.fillna(0.0)
        out["down_decay"] = down.groupby((~down).cumsum()).cumsum().astype(int)
        out["up_decay"] = up.groupby((~up).cumsum()).cumsum().astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapDecayParams,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        atr = indicators["atr"].to_numpy()
        ma = indicators["ma200"].to_numpy()
        down = indicators["down_decay"].to_numpy()
        up = indicators["up_decay"].to_numpy()
        position = 0
        entry = 0.0
        stop_distance = 0.0
        held = 0

        for i in range(len(data)):
            if i and decisions[i - 1] != position:
                position = int(decisions[i - 1])
                if position:
                    entry = float(opens[i])
                    stop_distance = params.stop_atr * float(atr[i - 1])
                    held = 0

            if position:
                held += 1
                stopped = (position == 1 and closes[i] < entry - stop_distance) or (
                    position == -1 and closes[i] > entry + stop_distance
                )
                decisions[i] = 0 if stopped or held >= 2 else position
            elif i >= self.warmup_bars(params) and ma[i] > 0 and atr[i] > 0:
                if closes[i] > ma[i] and down[i] >= params.streak_length - 1 and closes[i] > opens[i]:
                    decisions[i] = 1
                elif closes[i] < ma[i] and up[i] >= params.streak_length - 1 and closes[i] < opens[i]:
                    decisions[i] = -1

        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
