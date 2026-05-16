from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    dd_window: int = 5
    acf_window: int = 5
    dd_thresh: float = 0.01
    acf_neg_thresh: float = 0.15
    profit_target: float = 0.012
    time_stop: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778911298"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # ret needs 1 bar, ret.shift(1) needs 2, rolling(acf_window) needs acf_window more,
        # then acf_lag1.shift(1) needs 1 more => acf_window + 3 price bars total
        return max(params.dd_window, params.acf_window + 3) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        ret = close.pct_change()

        roll_max = close.rolling(params.dd_window).max()
        drawdown = (close - roll_max) / roll_max

        ret_lag1 = ret.shift(1)
        acf_lag1 = ret.rolling(params.acf_window).corr(ret_lag1)

        ind = pd.DataFrame(index=data.index)
        ind["drawdown"] = drawdown
        ind["acf_lag1"] = acf_lag1
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"]
        drawdown = indicators["drawdown"]
        acf_lag1 = indicators["acf_lag1"]

        acf_prev = acf_lag1.shift(1)
        in_drawdown = drawdown < -params.dd_thresh
        acf_was_neg = acf_prev < -params.acf_neg_thresh
        acf_recovering = acf_lag1 > acf_prev

        entry_cond = (in_drawdown & acf_was_neg & acf_recovering).fillna(False).values
        close_arr = close.values
        n = len(close_arr)

        signal_raw = np.zeros(n, dtype=np.int64)
        in_trade = False
        entry_price = 0.0
        bars_in_trade = 0

        for i in range(n):
            if in_trade:
                bars_in_trade += 1
                pnl_pct = (close_arr[i] - entry_price) / entry_price
                if pnl_pct >= params.profit_target or bars_in_trade >= params.time_stop:
                    in_trade = False
                else:
                    signal_raw[i] = 1

            if not in_trade and entry_cond[i]:
                signal_raw[i] = 1
                in_trade = True
                entry_price = close_arr[i]
                bars_in_trade = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = (
            pd.Series(signal_raw, index=data.index).shift(1).fillna(0).astype(int)
        )
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
