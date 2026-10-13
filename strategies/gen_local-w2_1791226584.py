from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AreaParams:
    window: int = 10
    threshold: float = 0.12


class GeneratedStrategy(BaseStrategy[AreaParams]):
    strategy_id = "gen_local-w2_1791226584"

    @classmethod
    def params_type(cls):
        return AreaParams

    def warmup_bars(self, params: AreaParams) -> int:
        return params.window + 1

    def indicators(self, data: pd.DataFrame, params: AreaParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["gap"] = np.log(data["open"] / data["close"].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["intraday"] = np.log(data["close"] / data["open"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ret = np.log(data["close"] / data["close"].shift(1)).replace([np.inf, -np.inf], np.nan)
        out["vol"] = ret.rolling(params.window, min_periods=params.window).std(ddof=0).fillna(0.0)

        x = out["gap"].cumsum()
        y = out["intraday"].cumsum()
        cumulative_area = (0.5 * (x.shift(1).fillna(0.0) * out["intraday"] - y.shift(1).fillna(0.0) * out["gap"])).cumsum()
        x0 = x.shift(params.window).fillna(0.0)
        y0 = y.shift(params.window).fillna(0.0)
        area = cumulative_area - cumulative_area.shift(params.window).fillna(0.0)
        area -= 0.5 * (x0 * (y - y0) - y0 * (x - x0))
        scale = (params.window * out["vol"].pow(2)).replace(0.0, np.nan)
        out["area"] = (area / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AreaParams,
    ) -> SignalFrame:
        signals = []
        state = 0
        for score in indicators["area"]:
            if state == 0:
                if score > params.threshold:
                    state = 1
                elif score < -params.threshold:
                    state = -1
            elif state == 1 and score < -params.threshold:
                state = 0
            elif state == -1 and score > params.threshold:
                state = 0
            signals.append(state)

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(signals, index=data.index, dtype=int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        annual_vol = indicators["vol"] * np.sqrt(252)
        df["size"] = (0.15 / annual_vol.replace(0.0, np.nan)).clip(lower=0.25, upper=1.0).fillna(1.0)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
