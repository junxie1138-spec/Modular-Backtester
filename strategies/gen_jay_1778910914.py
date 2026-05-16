from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    acf_window: int = 20
    accel_window: int = 5
    capacity_thresh: float = 0.20
    accel_thresh: float = 0.05
    hold_bars: int = 15


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778910914"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.acf_window + params.accel_window + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        ret = data["close"].pct_change()
        ret_lag1 = ret.shift(1)
        # Rolling lag-1 autocorrelation: measures momentum-persistence pressure
        acf = ret.rolling(params.acf_window).corr(ret_lag1)
        ind["acf"] = acf
        # Rate-of-change (acceleration) of the autocorrelation
        acf_roc = acf.diff(params.accel_window)
        ind["acf_roc"] = acf_roc
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = 1.0

        acf = indicators["acf"]
        acf_roc = indicators["acf_roc"]

        # Queue-overflow contrarian signals
        # Momentum queue full + now draining -> short
        raw_short = (acf > params.capacity_thresh) & (acf_roc < -params.accel_thresh)
        # Mean-reversion queue full + now draining -> long
        raw_long = (acf < -params.capacity_thresh) & (acf_roc > params.accel_thresh)

        # Two-bar confirmation: current AND previous bar must agree
        conf_short = raw_short & raw_short.shift(1).fillna(False)
        conf_long = raw_long & raw_long.shift(1).fillna(False)

        raw_signal = conf_long.astype(int) - conf_short.astype(int)

        # Fixed-bar exit: hold exactly hold_bars after entry, loop is required
        n = len(df)
        signal_arr = np.zeros(n, dtype=int)
        raw_arr = raw_signal.fillna(0).to_numpy(dtype=int)
        position = 0
        entry_bar = -1

        for i in range(n):
            if position != 0 and (i - entry_bar) >= params.hold_bars:
                position = 0
                entry_bar = -1
            if position == 0 and raw_arr[i] != 0:
                position = raw_arr[i]
                entry_bar = i
            signal_arr[i] = position

        df["signal"] = signal_arr
        # Mandatory 1-bar shift: decision at bar N executes at bar N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
