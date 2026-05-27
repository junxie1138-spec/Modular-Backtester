from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    range_lookback: int = 20
    spike_multiplier: float = 2.0
    refractory_bars: int = 5
    breakout_horizon: int = 20
    atr_period: int = 14
    atr_stop_mult: float = 3.0
    holding_bars_max: int = 30
    size_floor: float = 0.4
    size_ceiling: float = 1.0
    size_scale: float = 0.35


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1779648162"

    @classmethod
    def params_type(cls):
        return Params

    @classmethod
    def warmup_bars(cls, params: Params) -> int:
        return int(max(params.range_lookback, params.atr_period)) + 2

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        prev_close = close.shift(1)
        tr_components = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        )
        true_range = tr_components.max(axis=1)
        atr = true_range.rolling(
            window=params.atr_period, min_periods=params.atr_period
        ).mean()

        bar_range = high - low
        typical_range = bar_range.rolling(
            window=params.range_lookback, min_periods=params.range_lookback
        ).mean()

        denom = typical_range.replace(0.0, np.nan)
        spike_ratio = bar_range / denom

        is_spike = (spike_ratio >= params.spike_multiplier).fillna(False)

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["typical_range"] = typical_range
        out["spike_ratio"] = spike_ratio.fillna(0.0)
        out["is_spike"] = is_spike.astype(bool)
        return out

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        raw_signal = np.zeros(n, dtype=np.int64)
        raw_size = np.ones(n, dtype=np.float64)

        close_arr = data["close"].to_numpy(dtype=np.float64)
        high_arr = data["high"].to_numpy(dtype=np.float64)
        atr_arr = indicators["atr"].to_numpy(dtype=np.float64)
        spike_arr = indicators["is_spike"].to_numpy()
        ratio_arr = indicators["spike_ratio"].to_numpy(dtype=np.float64)

        STATE_IDLE = 0
        STATE_REFRACTORY = 1
        STATE_ARMED = 2
        STATE_IN_POS = 3

        state = STATE_IDLE
        spike_high = 0.0
        spike_magnitude = 1.0
        refractory_left = 0
        arm_clock = 0
        entry_close = 0.0
        entry_atr = 0.0
        bars_held = 0
        current_size = 1.0

        for i in range(n):
            atr_i = atr_arr[i]
            close_i = close_arr[i]
            high_i = high_arr[i]
            spike_i = bool(spike_arr[i])
            ratio_i = ratio_arr[i]

            if state == STATE_IN_POS:
                bars_held += 1
                stop_hit = (
                    not np.isnan(entry_atr)
                    and close_i < (entry_close - params.atr_stop_mult * entry_atr)
                )
                time_out = bars_held >= params.holding_bars_max
                if stop_hit or time_out:
                    raw_signal[i] = 0
                    raw_size[i] = 1.0
                    state = STATE_IDLE
                    spike_high = 0.0
                    spike_magnitude = 1.0
                    entry_close = 0.0
                    entry_atr = 0.0
                    bars_held = 0
                    current_size = 1.0
                else:
                    raw_signal[i] = 1
                    raw_size[i] = current_size

            elif state == STATE_IDLE:
                if spike_i and not np.isnan(atr_i):
                    spike_high = high_i
                    mag = ratio_i if not np.isnan(ratio_i) else 1.0
                    spike_magnitude = float(mag)
                    refractory_left = int(params.refractory_bars)
                    state = STATE_REFRACTORY
                raw_signal[i] = 0
                raw_size[i] = 1.0

            elif state == STATE_REFRACTORY:
                if high_i > spike_high:
                    spike_high = high_i
                if spike_i:
                    refractory_left = int(params.refractory_bars)
                    if not np.isnan(ratio_i):
                        spike_magnitude = max(spike_magnitude, float(ratio_i))
                refractory_left -= 1
                if refractory_left <= 0:
                    state = STATE_ARMED
                    arm_clock = int(params.breakout_horizon)
                raw_signal[i] = 0
                raw_size[i] = 1.0

            elif state == STATE_ARMED:
                if spike_i and not np.isnan(ratio_i):
                    spike_high = max(spike_high, high_i)
                    spike_magnitude = max(spike_magnitude, float(ratio_i))
                    refractory_left = int(params.refractory_bars)
                    state = STATE_REFRACTORY
                    raw_signal[i] = 0
                    raw_size[i] = 1.0
                elif not np.isnan(atr_i) and close_i > spike_high:
                    entry_close = close_i
                    entry_atr = atr_i
                    bars_held = 0
                    raw_pos = params.size_scale * spike_magnitude
                    current_size = float(
                        np.clip(raw_pos, params.size_floor, params.size_ceiling)
                    )
                    state = STATE_IN_POS
                    raw_signal[i] = 1
                    raw_size[i] = current_size
                else:
                    arm_clock -= 1
                    raw_signal[i] = 0
                    raw_size[i] = 1.0
                    if arm_clock <= 0:
                        state = STATE_IDLE
                        spike_high = 0.0
                        spike_magnitude = 1.0

        df = pd.DataFrame(index=data.index)
        signal_series = pd.Series(raw_signal, index=data.index)
        size_series = pd.Series(raw_size, index=data.index)

        df["signal"] = signal_series.shift(1).fillna(0).astype(int)
        df["size"] = size_series.shift(1).fillna(1.0).astype(float).clip(lower=1e-6)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
