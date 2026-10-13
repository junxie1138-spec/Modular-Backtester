from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VarianceRatioParams:
    vol_window: int = 10
    percentile_window: int = 126
    atr_window: int = 14
    entry_percentile: float = 0.85
    breakeven_gain: float = 0.006
    trail_atr: float = 0.9
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[VarianceRatioParams]):
    strategy_id = "gen_local-w2_1791226485"

    @classmethod
    def params_type(cls) -> type[VarianceRatioParams]:
        return VarianceRatioParams

    @classmethod
    def warmup_bars(cls, params: VarianceRatioParams) -> int:
        return max(params.vol_window + params.percentile_window + 2, params.atr_window)

    def indicators(self, data: pd.DataFrame, params: VarianceRatioParams) -> pd.DataFrame:
        daily = data["close"].pct_change()
        two_day = data["close"].pct_change(2)
        two_day_vol = two_day.rolling(params.vol_window).std().replace(0, np.nan)
        ratio = np.sqrt(2.0) * daily.rolling(params.vol_window).std() / two_day_vol
        threshold = ratio.rolling(params.percentile_window).quantile(params.entry_percentile)
        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_window).mean()
        return pd.DataFrame(
            {"ratio": ratio.fillna(0.0), "threshold": threshold.fillna(0.0), "atr": atr.fillna(0.0)},
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VarianceRatioParams,
    ) -> SignalFrame:
        entry_condition = (
            (data["close"] < data["close"].shift(1))
            & (indicators["threshold"] > 0)
            & (indicators["ratio"] >= indicators["threshold"])
            & (indicators["atr"] > 0)
        ).fillna(False)
        decisions = pd.Series(0, index=data.index, dtype=int)
        active = False
        pending_entry = False
        entry = 0.0
        stop = float("-inf")
        armed = False
        held = 0

        for i in range(len(data)):
            if pending_entry:
                active = True
                pending_entry = False
                entry = float(data["open"].iat[i])
                stop = float("-inf")
                armed = False
                held = 0

            if active:
                held += 1
                high = float(data["high"].iat[i])
                close = float(data["close"].iat[i])
                if high >= entry * (1.0 + params.breakeven_gain):
                    armed = True
                if armed:
                    stop = max(stop, entry, high - params.trail_atr * float(indicators["atr"].iat[i]))
                if close <= stop or held >= params.max_hold_bars:
                    active = False
                else:
                    decisions.iat[i] = 1
            elif bool(entry_condition.iat[i]):
                decisions.iat[i] = 1
                pending_entry = True

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
