from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ChangeParams:
    baseline_window: int = 20
    change_z: float = 1.5
    atr_window: int = 14
    trail_atr: float = 2.0
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[ChangeParams]):
    strategy_id = "gen_local_1791296915"

    @classmethod
    def params_type(cls) -> type[ChangeParams]:
        return ChangeParams

    @classmethod
    def warmup_bars(cls, params: ChangeParams) -> int:
        return max(params.baseline_window + 3, params.atr_window)

    def indicators(self, data: pd.DataFrame, params: ChangeParams) -> pd.DataFrame:
        ret = data["close"].pct_change()
        fast = ret.rolling(2, min_periods=2).mean()
        prior = ret.shift(2)
        baseline = prior.rolling(params.baseline_window, min_periods=params.baseline_window)
        score = ((fast - baseline.mean()) / baseline.std().replace(0, np.nan))
        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        return pd.DataFrame(
            {
                "return": ret.fillna(0.0),
                "change_score": score.replace([np.inf, -np.inf], np.nan).fillna(0.0),
                "atr": atr.fillna(0.0),
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ChangeParams,
    ) -> SignalFrame:
        ret = indicators["return"]
        score = indicators["change_score"]
        entry = (
            (ret > 0)
            & (ret.shift(1).fillna(0) > 0)
            & (score > params.change_z)
            & (score.shift(1).fillna(0) <= params.change_z)
            & (indicators["atr"] > 0)
        )
        signals = np.zeros(len(data), dtype=int)
        in_position = False
        held = 0
        peak = 0.0
        stop = 0.0
        for i in range(len(data)):
            close = float(data["close"].iloc[i])
            atr = float(indicators["atr"].iloc[i])
            if in_position:
                held += 1
                peak = max(peak, close) if held > 1 else close
                stop = max(stop, peak - params.trail_atr * atr) if held > 1 else peak - params.trail_atr * atr
                if close < stop or held >= params.max_hold:
                    in_position = False
                else:
                    signals[i] = 1
            elif bool(entry.iloc[i]):
                in_position = True
                held = 0
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
