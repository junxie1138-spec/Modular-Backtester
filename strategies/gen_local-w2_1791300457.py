from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakAnalogParams:
    analog_window: int = 504
    horizon: int = 15
    edge_threshold: float = 0.004
    atr_period: int = 14
    stop_atr: float = 2.5
    hold_bars: int = 18
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[StreakAnalogParams]):
    strategy_id = "gen_local-w2_1791300457"

    @classmethod
    def params_type(cls):
        return StreakAnalogParams

    def warmup_bars(self, params: StreakAnalogParams) -> int:
        return max(params.analog_window + params.horizon + 1, params.atr_period)

    def indicators(self, data: pd.DataFrame, params: StreakAnalogParams) -> pd.DataFrame:
        close = data["close"]
        ret = close.pct_change(fill_method=None).fillna(0.0)
        direction = np.sign(close.diff().fillna(0.0)).astype(int)
        streak = direction.groupby(direction.ne(direction.shift()).cumsum()).cumcount().add(1)
        compression = ret.abs().lt(ret.abs().shift(1)).astype(int)
        shape = (direction * (streak.clip(upper=4) + 3 * compression)).where(streak.ge(2), 0)

        realized = close.pct_change(params.horizon, fill_method=None)
        past_shape = shape.shift(params.horizon)
        edge = pd.Series(np.nan, index=data.index, dtype=float)
        for code in (-7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7):
            matched = past_shape.eq(code) & realized.notna()
            count = matched.rolling(params.analog_window, min_periods=params.analog_window).sum()
            total = realized.where(matched, 0.0).rolling(
                params.analog_window, min_periods=params.analog_window
            ).sum()
            edge = edge.mask(shape.eq(code), total.div(count.where(count.ge(12))))

        previous_close = close.shift(1)
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_period, min_periods=params.atr_period).mean()
        return pd.DataFrame({"edge": edge, "atr": atr}, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakAnalogParams,
    ) -> SignalFrame:
        edge = indicators["edge"]
        candidate = pd.Series(
            np.select(
                [edge.gt(params.edge_threshold), edge.lt(-params.edge_threshold)],
                [1, -1],
                default=0,
            ),
            index=data.index,
        )
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0)
        target = np.zeros(len(data), dtype=int)
        position = 0
        entry_stop = 0.0
        held = 0

        for i in range(len(data)):
            if i and target[i - 1] != position:
                position = int(target[i - 1])
                held = 0
                if position:
                    entry_stop = float(data["open"].iloc[i]) - position * params.stop_atr * float(indicators["atr"].iloc[i - 1])

            if position:
                held += 1
                if position * (float(data["close"].iloc[i]) - entry_stop) <= 0 or held >= params.hold_bars:
                    target[i] = 0
                else:
                    target[i] = position
            elif confirmed.iloc[i] and pd.notna(indicators["atr"].iloc[i]) and indicators["atr"].iloc[i] > 0:
                target[i] = int(confirmed.iloc[i])

        df = pd.DataFrame(index=data.index)
        df["signal"] = target
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = max(float(params.size), 1e-9)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
