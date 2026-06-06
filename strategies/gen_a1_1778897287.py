from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    lookback_n: int = 20
    short_window: int = 5
    entry_drop: float = 0.05
    atr_period: int = 14
    k_init_atr: float = 2.0
    k_atr: float = 2.5
    be_trigger: float = 0.04
    max_hold: int = 25


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778897287"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(max(params.lookback_n, 2 * params.short_window, params.atr_period)) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ret = close.pct_change()

        cum_ret = ret.rolling(params.lookback_n).sum()
        vel_recent = ret.rolling(params.short_window).sum()
        vel_prior = vel_recent.shift(params.short_window)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_period).mean()

        out = pd.DataFrame(index=data.index)
        out["cum_ret"] = cum_ret
        out["vel_recent"] = vel_recent
        out["vel_prior"] = vel_prior
        out["atr"] = atr
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)

        cum_ret = indicators["cum_ret"].to_numpy(dtype=float)
        vel_recent = indicators["vel_recent"].to_numpy(dtype=float)
        vel_prior = indicators["vel_prior"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        sig = np.zeros(n, dtype=np.int64)

        position = 0
        entry_price = 0.0
        stop = 0.0
        breakeven_done = False
        bars_held = 0

        for i in range(n):
            valid = (
                not np.isnan(cum_ret[i])
                and not np.isnan(vel_recent[i])
                and not np.isnan(vel_prior[i])
                and not np.isnan(atr[i])
                and atr[i] > 0.0
            )

            if position == 0:
                if valid:
                    # Primitive 1: deep multi-week close-to-close discount.
                    deep_discount = cum_ret[i] <= -params.entry_drop
                    # Primitive 2: shockwave front decelerating - the prior
                    # leg fell hard, the recent leg fell less hard.
                    shockwave_dissipating = (
                        vel_prior[i] < 0.0 and vel_recent[i] > vel_prior[i]
                    )
                    if deep_discount and shockwave_dissipating:
                        position = 1
                        entry_price = close[i]
                        stop = entry_price - params.k_init_atr * atr[i]
                        breakeven_done = False
                        bars_held = 0
                        sig[i] = 1
            else:
                bars_held += 1

                if not breakeven_done:
                    if high[i] >= entry_price * (1.0 + params.be_trigger):
                        if entry_price > stop:
                            stop = entry_price
                        breakeven_done = True

                if breakeven_done and not np.isnan(atr[i]):
                    trail = close[i] - params.k_atr * atr[i]
                    if trail > stop:
                        stop = trail

                exit_now = low[i] <= stop or bars_held >= params.max_hold

                if exit_now:
                    position = 0
                    entry_price = 0.0
                    stop = 0.0
                    breakeven_done = False
                    bars_held = 0
                    sig[i] = 0
                else:
                    sig[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
