from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalParams:
    downside_efficiency: float = 0.4
    atr_multiple: float = 2.0


class GeneratedStrategy(BaseStrategy[SeasonalParams]):
    strategy_id = "gen_local_1790994295"

    @classmethod
    def params_type(cls) -> type[SeasonalParams]:
        return SeasonalParams

    def warmup_bars(self, params: SeasonalParams) -> int:
        return 15

    def indicators(self, data: pd.DataFrame, params: SeasonalParams) -> pd.DataFrame:
        close = data["close"]
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        net = returns.rolling(3, min_periods=3).sum()
        noise = returns.abs().rolling(3, min_periods=3).sum()
        efficiency = net.div(noise.where(noise > 0))
        previous_close = close.shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(14, min_periods=14).mean()
        day = pd.Series(data.index.day, index=data.index)
        event = day.between(15, 18) & day.shift(1).lt(15)
        return pd.DataFrame(
            {"return": returns, "net_return": net, "absolute_return_sum": noise,
             "downside_efficiency": efficiency, "true_range": true_range,
             "atr": atr, "midmonth_event": event},
            index=data.index,
        )

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: SeasonalParams,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = (
            indicators["midmonth_event"]
            & indicators["downside_efficiency"].le(-params.downside_efficiency)
            & indicators["atr"].gt(0)
        ).fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and np.isfinite(atr) and atr > 0
            if active:
                age += 1
                if not valid:
                    active = False
                else:
                    # Entry fills on the first session after the entry decision.
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.atr_multiple * atr)
                    if close <= stop or age >= 4:
                        active = False
                signals[i] = int(active)
            elif i >= self.warmup_bars(params) - 1 and valid and entries[i]:
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
