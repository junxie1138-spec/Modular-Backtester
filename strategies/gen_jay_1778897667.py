from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    breakout_window: int = 20
    drawdown_window: int = 10
    percentile_window: int = 126
    arm_pct: float = 0.65
    fire_pct: float = 0.35
    atr_window: int = 14
    atr_mult: float = 2.0
    max_hold: int = 5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778897667"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.percentile_window + params.drawdown_window + params.atr_window + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        close = data["close"]
        high = data["high"]
        low = data["low"]

        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        roll_peak = close.rolling(params.drawdown_window, min_periods=params.drawdown_window).max()
        raw_dd = (roll_peak - close) / roll_peak.where(roll_peak > 0, np.nan)
        ind["drawdown"] = raw_dd

        ind["dd_rank"] = raw_dd.rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).apply(
            lambda x: float(np.sum(x[:-1] <= x[-1])) / max(len(x) - 1, 1),
            raw=True,
        )

        ind["roll_high"] = (
            close.rolling(params.breakout_window, min_periods=params.breakout_window)
            .max()
            .shift(1)
        )
        ind["roll_low"] = (
            close.rolling(params.breakout_window, min_periods=params.breakout_window)
            .min()
            .shift(1)
        )

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close_arr = data["close"].to_numpy(dtype=float)
        atr_arr = indicators["atr"].to_numpy(dtype=float)
        dd_rank_arr = indicators["dd_rank"].to_numpy(dtype=float)
        roll_high_arr = indicators["roll_high"].to_numpy(dtype=float)
        roll_low_arr = indicators["roll_low"].to_numpy(dtype=float)

        n = len(close_arr)
        raw_signal = np.zeros(n, dtype=np.int64)

        position = 0
        entry_price = 0.0
        entry_atr = 0.0
        bars_held = 0
        armed_long = False
        armed_short = False

        for i in range(n):
            rank = dd_rank_arr[i]
            c = close_arr[i]
            a = atr_arr[i]
            rh = roll_high_arr[i]
            rl = roll_low_arr[i]

            if np.isnan(rank) or np.isnan(a) or np.isnan(rh) or np.isnan(rl):
                continue

            if position == 1:
                bars_held += 1
                if c < entry_price - params.atr_mult * entry_atr or bars_held >= params.max_hold:
                    raw_signal[i] = 0
                    position = 0
                    bars_held = 0
                    armed_long = False
                else:
                    raw_signal[i] = 1
                continue

            if position == -1:
                bars_held += 1
                if c > entry_price + params.atr_mult * entry_atr or bars_held >= params.max_hold:
                    raw_signal[i] = 0
                    position = 0
                    bars_held = 0
                    armed_short = False
                else:
                    raw_signal[i] = -1
                continue

            if rank >= params.arm_pct:
                armed_long = True
            if rank <= (1.0 - params.arm_pct):
                armed_short = True

            if armed_long and rank <= params.fire_pct and c > rh:
                raw_signal[i] = 1
                position = 1
                entry_price = c
                entry_atr = a
                bars_held = 0
                armed_long = False
                armed_short = False
            elif armed_short and rank >= (1.0 - params.fire_pct) and c < rl:
                raw_signal[i] = -1
                position = -1
                entry_price = c
                entry_atr = a
                bars_held = 0
                armed_short = False
                armed_long = False

        idx = data.index
        sig = pd.Series(raw_signal, index=idx, dtype=np.int64)
        size = pd.Series(np.ones(n, dtype=float), index=idx)

        sig = sig.shift(1).fillna(0).astype(int)
        size = size.shift(1).fillna(1.0)

        df = pd.DataFrame({"signal": sig, "size": size}, index=idx)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
