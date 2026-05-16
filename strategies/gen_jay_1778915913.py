from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 20
    atr_mult: float = 2.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778915913"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return params.lookback + 15

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        high = data["high"]
        low = data["low"]
        close = data["close"]

        daily_range = high - low
        avg_range = daily_range.rolling(params.lookback).mean()
        range_ratio = daily_range / avg_range.where(avg_range > 0)

        rolling_high_prior = high.shift(1).rolling(params.lookback).max()
        rolling_low_prior = low.shift(1).rolling(params.lookback).min()

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14).mean()

        ind["range_ratio"] = range_ratio
        ind["rolling_high_prior"] = rolling_high_prior
        ind["rolling_low_prior"] = rolling_low_prior
        ind["atr"] = atr
        ind["close"] = close

        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = indicators["close"].values
        rolling_high_prior = indicators["rolling_high_prior"].values
        rolling_low_prior = indicators["rolling_low_prior"].values
        range_ratio = indicators["range_ratio"].values
        atr = indicators["atr"].values

        n = len(close)
        raw_signal = np.zeros(n, dtype=int)

        RANGE_SHOCK_THRESH = 1.5

        position = 0
        hwm = np.nan
        lwm = np.nan

        for i in range(n):
            if (
                np.isnan(rolling_high_prior[i])
                or np.isnan(rolling_low_prior[i])
                or np.isnan(range_ratio[i])
                or np.isnan(atr[i])
            ):
                raw_signal[i] = 0
                position = 0
                hwm = np.nan
                lwm = np.nan
                continue

            c = close[i]
            stop_fired = False

            if position == 1:
                if c > hwm:
                    hwm = c
                if c < hwm - params.atr_mult * atr[i]:
                    position = 0
                    hwm = np.nan
                    stop_fired = True
            elif position == -1:
                if c < lwm:
                    lwm = c
                if c > lwm + params.atr_mult * atr[i]:
                    position = 0
                    lwm = np.nan
                    stop_fired = True

            if position == 0 and not stop_fired:
                shockwave = range_ratio[i] > RANGE_SHOCK_THRESH
                if shockwave:
                    if c > rolling_high_prior[i]:
                        position = 1
                        hwm = c
                    elif c < rolling_low_prior[i]:
                        position = -1
                        lwm = c

            raw_signal[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw_signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
