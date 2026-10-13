from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    regime_window: int = 10
    streak_trigger: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790999520"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(2, int(params.regime_window), int(params.streak_trigger))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        window = max(2, int(params.regime_window))
        trigger = max(1, int(params.streak_trigger))
        body = data["close"] - data["open"]
        side = np.sign(body).fillna(0).astype(int)
        groups = side.ne(side.shift(1)).cumsum()
        streak = side.ne(0).astype(int).groupby(groups).cumsum().clip(upper=trigger)
        span = (
            (data["high"] - data["low"])
            / data["close"].replace(0, np.nan)
        ).where(data["high"] > data["low"])
        volume = data["volume"].where(data["volume"] > 0)
        range_mean = span.rolling(window, min_periods=window).mean()
        volume_mean = volume.rolling(window, min_periods=window).mean()
        density = (
            (volume / volume_mean) / (span / range_mean)
        ).replace([np.inf, -np.inf], np.nan)
        ready = density.notna() & body.notna() & data["close"].gt(0)
        crowded = density.gt(1.0)
        direction = side.where(
            ~(crowded & streak.ge(trigger)), -side
        ).where(ready, 0).astype(int)
        return pd.DataFrame(
            {
                "body": body.fillna(0.0),
                "side": side,
                "streak": streak,
                "range_fraction": span.fillna(0.0),
                "range_mean": range_mean.fillna(0.0),
                "volume_mean": volume_mean.fillna(0.0),
                "participation_density": density.fillna(0.0),
                "crowded": crowded,
                "ready": ready,
                "entry_direction": direction,
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        state = 0
        targets = np.zeros(len(data), dtype=int)
        for i, desired in enumerate(
            indicators["entry_direction"].to_numpy(dtype=int)
        ):
            if state == 0:
                state = int(desired)
            elif desired == -state:
                # An opposing entry condition exits to flat for this bar.
                state = 0
            # Neutral conditions preserve an existing position.
            targets[i] = state
        df = pd.DataFrame(
            {"signal": targets, "size": 1.0}, index=data.index
        )
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
