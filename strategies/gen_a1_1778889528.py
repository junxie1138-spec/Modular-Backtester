from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownBreakoutParams:
    peak_window: int = 120
    breakout_window: int = 25
    depth_window: int = 40
    min_depth: float = 0.04
    max_depth: float = 0.18
    atr_window: int = 14
    atr_init_mult: float = 2.0
    breakeven_pct: float = 0.03
    k_atr: float = 3.0
    max_hold_bars: int = 20
    base_size: float = 0.4
    size_span: float = 0.6


class GeneratedStrategy(BaseStrategy[DrawdownBreakoutParams]):
    strategy_id = "gen_a1_1778889528"

    @classmethod
    def params_type(cls):
        return DrawdownBreakoutParams

    @staticmethod
    def warmup_bars(params: DrawdownBreakoutParams) -> int:
        return (
            params.peak_window
            + max(params.breakout_window, params.depth_window)
            + params.atr_window
            + 5
        )

    @staticmethod
    def indicators(data: pd.DataFrame, params: DrawdownBreakoutParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        peak = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        dd = close / peak - 1.0

        dd_high = dd.rolling(
            params.breakout_window, min_periods=params.breakout_window
        ).max()
        trough_dd = dd.rolling(
            params.depth_window, min_periods=params.depth_window
        ).min()

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        out = pd.DataFrame(index=data.index)
        out["dd"] = dd
        out["dd_prev"] = dd.shift(1)
        out["dd_high"] = dd_high
        out["trough_dd"] = trough_dd
        out["atr"] = atr
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        dd = indicators["dd"].to_numpy(dtype=float)
        dd_prev = indicators["dd_prev"].to_numpy(dtype=float)
        dd_high = indicators["dd_high"].to_numpy(dtype=float)
        trough_dd = indicators["trough_dd"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        span = params.max_depth - params.min_depth
        if span <= 0.0:
            span = 1e-9

        in_pos = False
        entry_price = 0.0
        stop = 0.0
        highest = 0.0
        breakeven = False
        hold = 0
        pos_size = 1.0

        for i in range(n):
            if not in_pos:
                valid = (
                    not np.isnan(dd[i])
                    and not np.isnan(dd_prev[i])
                    and not np.isnan(dd_high[i])
                    and not np.isnan(trough_dd[i])
                    and not np.isnan(atr[i])
                )
                if valid:
                    fresh_high = (
                        dd[i] >= dd_high[i] - 1e-12 and dd[i] > dd_prev[i]
                    )
                    still_down = dd[i] < 0.0
                    depth = -trough_dd[i]
                    capacity_ok = (
                        depth >= params.min_depth and depth <= params.max_depth
                    )
                    if fresh_high and still_down and capacity_ok and atr[i] > 0.0:
                        strength = (depth - params.min_depth) / span
                        if strength < 0.0:
                            strength = 0.0
                        elif strength > 1.0:
                            strength = 1.0
                        pos_size = params.base_size + params.size_span * strength
                        in_pos = True
                        entry_price = close[i]
                        highest = close[i]
                        stop = entry_price - params.atr_init_mult * atr[i]
                        breakeven = False
                        hold = 0
                        signal[i] = 1
                        size[i] = pos_size
                continue

            hold += 1
            if close[i] > highest:
                highest = close[i]
            if not breakeven and close[i] >= entry_price * (1.0 + params.breakeven_pct):
                breakeven = True

            a = atr[i] if not np.isnan(atr[i]) else 0.0
            if breakeven:
                candidate = highest - params.k_atr * a
                if candidate < entry_price:
                    candidate = entry_price
            else:
                candidate = entry_price - params.atr_init_mult * a
            if candidate > stop:
                stop = candidate

            exit_now = close[i] <= stop or hold >= params.max_hold_bars
            if exit_now:
                in_pos = False
                signal[i] = 0
                size[i] = 1.0
            else:
                signal[i] = 1
                size[i] = pos_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
