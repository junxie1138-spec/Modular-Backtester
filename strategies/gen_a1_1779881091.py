from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    pop_window: int = 30
    extreme_window: int = 20
    range_window: int = 20
    atr_window: int = 14
    asymmetry_threshold: float = 0.35
    range_expansion_mult: float = 1.2
    max_hold_bars: int = 18
    breakeven_pct: float = 0.015
    trail_atr_mult: float = 2.0
    size_exponent: float = 1.0
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1779881091"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.pop_window + params.extreme_window,
                       params.range_window,
                       params.atr_window) + 5)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        high = data["high"]
        low = data["low"]
        close = data["close"]

        prev_close = close.shift(1)
        tr_a = (high - low).abs()
        tr_b = (high - prev_close).abs()
        tr_c = (low - prev_close).abs()
        tr = pd.concat([tr_a, tr_b, tr_c], axis=1).max(axis=1)
        out["atr"] = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        prior_max = high.shift(1).rolling(params.extreme_window, min_periods=params.extreme_window).max()
        prior_min = low.shift(1).rolling(params.extreme_window, min_periods=params.extreme_window).min()
        predator = (high > prior_max).astype(float)
        prey = (low < prior_min).astype(float)

        pred_count = predator.rolling(params.pop_window, min_periods=params.pop_window).sum()
        prey_count = prey.rolling(params.pop_window, min_periods=params.pop_window).sum()

        asymmetry = (pred_count - prey_count) / (pred_count + prey_count + 1.0)
        out["asymmetry"] = asymmetry

        bar_range = (high - low)
        range_med = bar_range.shift(1).rolling(params.range_window, min_periods=params.range_window).median()
        out["range_ratio"] = bar_range / range_med.replace(0.0, np.nan)

        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        asym = indicators["asymmetry"].to_numpy(dtype=float)
        range_ratio = indicators["range_ratio"].to_numpy(dtype=float)

        n = len(data)
        raw_signal = np.zeros(n, dtype=np.int64)
        size = np.zeros(n, dtype=np.float64)

        position = 0
        entry_price = 0.0
        stop_price = 0.0
        breakeven_armed = False
        bars_held = 0
        entry_size = 0.0

        for i in range(n):
            inputs_ok = (np.isfinite(atr[i]) and np.isfinite(asym[i]) and np.isfinite(range_ratio[i]) and atr[i] > 0.0)

            if position == 0:
                if not inputs_ok:
                    continue
                entry_long = (asym[i] > params.asymmetry_threshold) and (range_ratio[i] > params.range_expansion_mult)
                entry_short = (asym[i] < -params.asymmetry_threshold) and (range_ratio[i] > params.range_expansion_mult)

                if entry_long:
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - params.trail_atr_mult * atr[i]
                    breakeven_armed = False
                    bars_held = 0
                    entry_size = params.base_size * (abs(asym[i]) ** params.size_exponent)
                    if not np.isfinite(entry_size) or entry_size <= 0.0:
                        entry_size = params.base_size
                    raw_signal[i] = 1
                    size[i] = entry_size
                elif entry_short:
                    position = -1
                    entry_price = close[i]
                    stop_price = entry_price + params.trail_atr_mult * atr[i]
                    breakeven_armed = False
                    bars_held = 0
                    entry_size = params.base_size * (abs(asym[i]) ** params.size_exponent)
                    if not np.isfinite(entry_size) or entry_size <= 0.0:
                        entry_size = params.base_size
                    raw_signal[i] = -1
                    size[i] = entry_size
            else:
                bars_held += 1
                if position == 1:
                    if inputs_ok:
                        gain_pct = (close[i] - entry_price) / entry_price if entry_price > 0 else 0.0
                        if (not breakeven_armed) and gain_pct >= params.breakeven_pct:
                            if entry_price > stop_price:
                                stop_price = entry_price
                            breakeven_armed = True
                        if breakeven_armed:
                            trail = close[i] - params.trail_atr_mult * atr[i]
                            if trail > stop_price:
                                stop_price = trail
                    exit_stop = low[i] <= stop_price
                    exit_time = bars_held >= params.max_hold_bars
                    if exit_stop or exit_time:
                        position = 0
                        raw_signal[i] = 0
                        size[i] = 0.0
                        entry_size = 0.0
                        breakeven_armed = False
                    else:
                        raw_signal[i] = 1
                        size[i] = entry_size
                else:
                    if inputs_ok:
                        gain_pct = (entry_price - close[i]) / entry_price if entry_price > 0 else 0.0
                        if (not breakeven_armed) and gain_pct >= params.breakeven_pct:
                            if entry_price < stop_price:
                                stop_price = entry_price
                            breakeven_armed = True
                        if breakeven_armed:
                            trail = close[i] + params.trail_atr_mult * atr[i]
                            if trail < stop_price:
                                stop_price = trail
                    exit_stop = high[i] >= stop_price
                    exit_time = bars_held >= params.max_hold_bars
                    if exit_stop or exit_time:
                        position = 0
                        raw_signal[i] = 0
                        size[i] = 0.0
                        entry_size = 0.0
                        breakeven_armed = False
                    else:
                        raw_signal[i] = -1
                        size[i] = entry_size

        df = pd.DataFrame({"signal": raw_signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.0)
        df.loc[df["size"] <= 0.0, "size"] = params.base_size

        return SignalFrame(data=df, signal_column="signal", size_column="size")
