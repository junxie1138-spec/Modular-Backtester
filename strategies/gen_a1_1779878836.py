from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EpidemicVolumeParams:
    window: int = 21
    atr_mult: float = 3.0


class GeneratedStrategy(BaseStrategy):
    strategy_id = "gen_a1_1779878836"

    @classmethod
    def params_type(cls):
        return EpidemicVolumeParams

    @classmethod
    def warmup_bars(cls, params):
        return max(int(params.window), 63) + 25

    @classmethod
    def indicators(cls, data, params):
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        ret = close.pct_change()
        vol_median = volume.rolling(63, min_periods=20).median()
        infected = ((ret > 0.0) & (volume > vol_median)).astype(float)
        infected_fraction = infected.rolling(
            int(params.window), min_periods=int(params.window)
        ).mean()

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()

        realized_vol = ret.rolling(20, min_periods=20).std() * np.sqrt(252.0)

        return pd.DataFrame(
            {
                "infected_fraction": infected_fraction,
                "atr": atr,
                "realized_vol": realized_vol,
                "close_ref": close,
            },
            index=data.index,
        )

    @classmethod
    def generate_signals(cls, data, indicators, ctx, params):
        close = indicators["close_ref"].to_numpy(dtype=float)
        frac = indicators["infected_fraction"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        rvol = indicators["realized_vol"].to_numpy(dtype=float)

        n = len(close)
        raw = np.zeros(n, dtype=np.int64)

        threshold = 0.5
        atr_mult = float(params.atr_mult)

        in_position = False
        in_trade_high = -np.inf

        for i in range(n):
            f = frac[i]
            a = atr[i]
            c = close[i]

            if np.isnan(c):
                in_position = False
                in_trade_high = -np.inf
                continue

            if not in_position:
                if i == 0:
                    continue
                f_prev = frac[i - 1]
                if (
                    not np.isnan(f)
                    and not np.isnan(f_prev)
                    and not np.isnan(a)
                    and f_prev <= threshold
                    and f > threshold
                ):
                    in_position = True
                    in_trade_high = c
                    raw[i] = 1
            else:
                if c > in_trade_high:
                    in_trade_high = c
                if np.isnan(a):
                    raw[i] = 1
                    continue
                stop_level = in_trade_high - atr_mult * a
                if c <= stop_level:
                    in_position = False
                    in_trade_high = -np.inf
                    raw[i] = 0
                else:
                    raw[i] = 1

        target_vol = 0.15
        rv_safe = np.where(
            np.isnan(rvol) | (rvol <= 1e-8), target_vol, rvol
        )
        size_arr = np.clip(target_vol / rv_safe, 0.1, 2.0)

        sig_series = (
            pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        )
        size_series = pd.Series(size_arr, index=data.index).astype(float)
        size_series = size_series.clip(lower=0.01).fillna(1.0)

        df = pd.DataFrame(
            {"signal": sig_series, "size": size_series}, index=data.index
        )
        return SignalFrame(data=df, signal_column="signal", size_column="size")
