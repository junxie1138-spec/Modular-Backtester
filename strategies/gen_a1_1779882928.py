from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_window: int = 60
    zscore_window: int = 60
    infected_threshold: float = -1.5
    infected_fraction_window: int = 30
    saturation_threshold: float = 0.35
    recovery_zscore: float = -0.25
    saturation_lookback: int = 15
    atr_window: int = 14
    atr_multiplier: float = 2.5
    max_hold_bars: int = 25


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1779882928"

    @classmethod
    def params_type(cls):
        return Params

    @classmethod
    def warmup_bars(cls, params: Params) -> int:
        return max(
            params.ma_window,
            params.zscore_window,
            params.infected_fraction_window,
            params.atr_window,
            params.saturation_lookback,
        ) + 2

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        ma = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        sd = close.rolling(params.zscore_window, min_periods=params.zscore_window).std(ddof=0)
        sd_safe = sd.where(sd > 0.0, np.nan)
        z = (close - ma) / sd_safe

        infected = (z < params.infected_threshold).astype(float)
        infected = infected.where(z.notna(), np.nan)
        infected_fraction = infected.rolling(
            params.infected_fraction_window,
            min_periods=params.infected_fraction_window,
        ).mean()

        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        peak_infected = infected_fraction.rolling(
            params.saturation_lookback,
            min_periods=1,
        ).max()

        ind = pd.DataFrame(
            {
                "z": z,
                "infected_fraction": infected_fraction,
                "peak_infected": peak_infected,
                "atr": atr,
            },
            index=data.index,
        )
        return ind

    def generate_signals(self, data, indicators, ctx, params):
        z = indicators["z"].to_numpy(copy=False)
        peak = indicators["peak_infected"].to_numpy(copy=False)
        atr = indicators["atr"].to_numpy(copy=False)
        close = data["close"].to_numpy(copy=False).astype(float)
        n = close.shape[0]

        valid = ~(np.isnan(z) | np.isnan(peak) | np.isnan(atr))
        armed = np.zeros(n, dtype=bool)
        for i in range(n):
            if not valid[i]:
                continue
            if peak[i] >= params.saturation_threshold and z[i] >= params.recovery_zscore:
                armed[i] = True

        raw_signal = np.zeros(n, dtype=np.int64)
        in_pos = False
        entry_idx = -1
        hwm = -np.inf

        for i in range(n):
            if not valid[i]:
                continue
            if in_pos:
                if close[i] > hwm:
                    hwm = close[i]
                stop_level = hwm - params.atr_multiplier * atr[i]
                hold = i - entry_idx
                exit_now = (close[i] <= stop_level) or (hold >= params.max_hold_bars)
                if exit_now:
                    raw_signal[i] = 0
                    in_pos = False
                    entry_idx = -1
                    hwm = -np.inf
                else:
                    raw_signal[i] = 1
            else:
                if i >= 1 and armed[i] and armed[i - 1]:
                    in_pos = True
                    entry_idx = i
                    hwm = close[i]
                    raw_signal[i] = 1

        sig = pd.Series(raw_signal, index=data.index)
        sig = sig.shift(1).fillna(0).astype(int)
        size = pd.Series(1.0, index=data.index, dtype=float)
        df = pd.DataFrame({"signal": sig, "size": size}, index=data.index)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
