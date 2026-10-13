from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownRegimeParams:
    depth: float = 0.04
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[DrawdownRegimeParams]):
    strategy_id = "gen_local_1791225484"

    @classmethod
    def params_type(cls) -> type[DrawdownRegimeParams]:
        return DrawdownRegimeParams

    @staticmethod
    def warmup_bars(params: DrawdownRegimeParams) -> int:
        return 64

    def indicators(self, data: pd.DataFrame, params: DrawdownRegimeParams) -> pd.DataFrame:
        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(14, min_periods=14).mean()
        long_atr = true_range.rolling(63, min_periods=63).mean()
        peak = data["close"].rolling(63, min_periods=63).max()

        result = pd.DataFrame(index=data.index)
        result["drawdown"] = (data["close"] / peak - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result["atr"] = atr.fillna(0.0)
        result["stressed"] = (atr / long_atr.replace(0.0, np.nan) > 1.25).fillna(False)
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownRegimeParams,
    ) -> SignalFrame:
        drawdown = indicators["drawdown"]
        prior_drawdown = drawdown.shift(1)
        stressed = indicators["stressed"]
        entry = (
            (~stressed & (drawdown <= -params.depth) & (prior_drawdown > -params.depth))
            | (stressed & (drawdown >= -2.0 * params.depth) & (prior_drawdown < -2.0 * params.depth))
        ).fillna(False)

        target = pd.Series(0, index=data.index, dtype=int)
        entry_bar = -1
        fixed_stop = 0.0
        for i in range(len(data)):
            if i > 0 and target.iat[i - 1] == 1:
                if entry_bar < 0:
                    entry_bar = i
                    fixed_stop = float(data["open"].iat[i]) - params.stop_atr * float(indicators["atr"].iat[i - 1])
                if float(data["close"].iat[i]) < fixed_stop or i - entry_bar + 1 >= 4:
                    entry_bar = -1
                else:
                    target.iat[i] = 1
            elif entry.iat[i] and indicators["atr"].iat[i] > 0:
                target.iat[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = target
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
