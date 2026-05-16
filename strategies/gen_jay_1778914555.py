from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    dd_window: int = 60
    er_window: int = 20
    arm_dd: float = 0.04
    disarm_dd: float = 0.01
    er_min: float = 0.25
    confirm_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778914555"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.dd_window + params.er_window + params.confirm_bars + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]

        rolling_high = close.rolling(params.dd_window, min_periods=params.dd_window).max()
        dd = (close - rolling_high) / rolling_high

        dd_grad = dd.diff()

        net_disp = (close - close.shift(params.er_window)).abs()
        path_len = (
            close.diff().abs()
            .rolling(params.er_window, min_periods=params.er_window)
            .sum()
        )
        er = (net_disp / path_len.replace(0, np.nan)).clip(0.0, 1.0)

        ind = pd.DataFrame(index=data.index)
        ind["dd"] = dd
        ind["dd_grad"] = dd_grad
        ind["er"] = er
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        dd = indicators["dd"].values
        dd_grad = indicators["dd_grad"].values
        er = indicators["er"].values

        n = len(data)
        raw_signal = np.zeros(n, dtype=int)

        armed = False
        confirm_count = 0
        in_position = False

        for i in range(n):
            dd_i = dd[i]
            grad_i = dd_grad[i]
            er_i = er[i]

            if np.isnan(dd_i) or np.isnan(grad_i) or np.isnan(er_i):
                raw_signal[i] = 1 if in_position else 0
                continue

            if not armed and dd_i < -params.arm_dd:
                armed = True

            if armed and dd_i > -params.disarm_dd:
                armed = False
                confirm_count = 0
                in_position = False

            entry_ok = armed and (grad_i > 0.0) and (er_i >= params.er_min)

            if not in_position:
                if entry_ok:
                    confirm_count += 1
                    if confirm_count >= params.confirm_bars:
                        in_position = True
                else:
                    confirm_count = 0
            else:
                if not entry_ok:
                    in_position = False
                    confirm_count = 0

            raw_signal[i] = 1 if in_position else 0

        df = pd.DataFrame(
            {"signal": raw_signal, "size": np.ones(n, dtype=float)},
            index=data.index,
        )
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
