from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    dd_window: int = 60
    ac_window: int = 20
    dd_entry_threshold: float = 0.03
    ac_threshold: float = 0.10
    ma_window: int = 200
    atr_window: int = 14
    atr_mult: float = 3.0
    size_gain: float = 1.0
    depth_cap: float = 0.10
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1778893156"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(
            params.ma_window,
            params.dd_window + params.ac_window + 2,
            params.atr_window,
        ) + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Drawdown depth relative to a rolling high-water close.
        roll_max = close.rolling(params.dd_window, min_periods=params.dd_window).max()
        drawdown = close / roll_max - 1.0

        # Lag-1 autocorrelation of the drawdown-increment series.
        # Negative => deepening tends to be followed by recovery (jam dissipating).
        dd_increment = drawdown.diff()
        dd_autocorr = dd_increment.rolling(
            params.ac_window, min_periods=params.ac_window
        ).corr(dd_increment.shift(1))

        # 200-day regime filter.
        ma = close.rolling(params.ma_window, min_periods=params.ma_window).mean()

        # Average true range for the trailing stop.
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        ind = pd.DataFrame(index=data.index)
        ind["drawdown"] = drawdown
        ind["dd_autocorr"] = dd_autocorr
        ind["ma"] = ma
        ind["atr"] = atr
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        drawdown = indicators["drawdown"].to_numpy(dtype=float)
        autocorr = indicators["dd_autocorr"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        n = len(close)
        raw_signal = np.zeros(n, dtype=int)
        size_arr = np.ones(n, dtype=float)

        depth_cap = params.depth_cap if params.depth_cap > 0.0 else 1.0

        position = 0
        high_water = 0.0
        bars_held = 0
        entry_size = 1.0

        for i in range(n):
            if position == 0:
                ready = (
                    np.isfinite(drawdown[i])
                    and np.isfinite(autocorr[i])
                    and np.isfinite(ma[i])
                    and np.isfinite(atr[i])
                )
                enter = (
                    ready
                    and drawdown[i] <= -params.dd_entry_threshold
                    and autocorr[i] <= -params.ac_threshold
                    and close[i] > ma[i]
                )
                if enter:
                    position = 1
                    high_water = close[i]
                    bars_held = 0
                    depth = min(-drawdown[i], depth_cap)
                    frac = depth / depth_cap
                    entry_size = 1.0 + params.size_gain * frac
                    raw_signal[i] = 1
                    size_arr[i] = entry_size
            else:
                bars_held += 1
                if close[i] > high_water:
                    high_water = close[i]
                stop_level = high_water - params.atr_mult * atr[i]
                stop_hit = np.isfinite(stop_level) and close[i] < stop_level
                time_out = bars_held >= params.max_hold_bars
                if stop_hit or time_out:
                    position = 0
                    raw_signal[i] = 0
                    size_arr[i] = 1.0
                else:
                    raw_signal[i] = 1
                    size_arr[i] = entry_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw_signal
        df["size"] = size_arr
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
