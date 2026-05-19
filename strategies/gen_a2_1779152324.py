from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    ac_window: int = 20
    ac_threshold: float = 0.10
    dd_window: int = 40
    dd_min: float = 0.03
    dd_max: float = 0.20
    exit_bars: int = 2


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a2_1779152324"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(max(params.ac_window, params.dd_window)) + 3

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        ret = close.pct_change()

        ac_window = max(int(params.ac_window), 3)
        dd_window = max(int(params.dd_window), 2)

        # Rolling lag-1 autocorrelation of daily returns (the elastic/plastic gauge).
        acf1 = ret.rolling(ac_window).corr(ret.shift(1))

        # Drawdown depth from a rolling peak (zero or negative).
        peak = close.rolling(dd_window, min_periods=1).max()
        drawdown = close / peak - 1.0

        # Drawdown shrinking (price recovering toward the peak).
        dd_improving = (drawdown > drawdown.shift(1)).astype(float)

        out = pd.DataFrame(index=data.index)
        out["acf1"] = acf1
        out["drawdown"] = drawdown
        out["dd_improving"] = dd_improving
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)

        acf1 = indicators["acf1"]
        drawdown = indicators["drawdown"]
        dd_improving = indicators["dd_improving"].fillna(0.0) > 0.5

        dd_min = abs(float(params.dd_min))
        dd_max = abs(float(params.dd_max))
        if dd_max < dd_min:
            dd_max = dd_min

        # Elastic regime: returns negatively autocorrelated -> dips snap back.
        elastic = (acf1 < -abs(float(params.ac_threshold))).fillna(False)
        # Moderate, non-catastrophic drawdown.
        moderate_dd = ((drawdown <= -dd_min) & (drawdown >= -dd_max)).fillna(False)
        # Two-bar confirmation: drawdown shrank on this bar and the prior bar.
        confirm = (dd_improving & dd_improving.shift(1).fillna(False))

        raw_entry = (elastic & moderate_dd & confirm).fillna(False).to_numpy()

        n = len(df)
        exit_bars = max(int(params.exit_bars), 1)
        signal = np.zeros(n, dtype=int)
        hold = 0
        for i in range(n):
            if hold > 0:
                signal[i] = 1
                hold -= 1
            elif raw_entry[i]:
                signal[i] = 1
                hold = exit_bars - 1

        df["signal"] = signal
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
