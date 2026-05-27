from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TidalAntinodeParams:
    envelope_window: int = 20
    short_range_window: int = 5
    long_range_window: int = 60
    amplitude_threshold: float = 1.0
    relpos_upper: float = 0.30
    relpos_lower: float = 0.05
    hold_bars: int = 15
    size_floor: float = 0.40
    size_ceiling: float = 1.00


class GeneratedStrategy(BaseStrategy[TidalAntinodeParams]):
    strategy_id = "gen_a1_1779648980"

    @classmethod
    def params_type(cls):
        return TidalAntinodeParams

    def warmup_bars(self, params: TidalAntinodeParams) -> int:
        return int(max(params.long_range_window, params.envelope_window, params.short_range_window)) + 2

    def indicators(self, data: pd.DataFrame, params: TidalAntinodeParams) -> pd.DataFrame:
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        bar_range = (high - low).clip(lower=0.0)

        env_high = high.rolling(params.envelope_window, min_periods=params.envelope_window).max()
        env_low = low.rolling(params.envelope_window, min_periods=params.envelope_window).min()
        env_span = (env_high - env_low).replace(0.0, np.nan)
        rel_pos = (close - env_low) / env_span

        short_range_mean = bar_range.rolling(params.short_range_window, min_periods=params.short_range_window).mean()
        long_range_mean = bar_range.rolling(params.long_range_window, min_periods=params.long_range_window).mean().replace(0.0, np.nan)
        tidal_amp = short_range_mean / long_range_mean

        return pd.DataFrame(
            {
                "rel_pos": rel_pos,
                "tidal_amp": tidal_amp,
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TidalAntinodeParams,
    ) -> SignalFrame:
        n = len(data)
        rel_pos = indicators["rel_pos"].to_numpy(dtype=float)
        tidal_amp = indicators["tidal_amp"].to_numpy(dtype=float)

        signal = np.zeros(n, dtype=np.int64)
        size = np.ones(n, dtype=float)

        bars_remaining = 0
        active_size = 0.0

        band_width = max(params.relpos_upper - params.relpos_lower, 1e-9)
        size_range = max(params.size_ceiling - params.size_floor, 0.0)
        floor = params.size_floor if params.size_floor > 0.0 else 1e-6
        hold = int(params.hold_bars) if params.hold_bars > 0 else 1

        for i in range(n):
            if bars_remaining > 0:
                signal[i] = 1
                size[i] = active_size
                bars_remaining -= 1
                if bars_remaining == 0:
                    active_size = 0.0
                continue

            rp = rel_pos[i]
            ta = tidal_amp[i]
            if not np.isfinite(rp) or not np.isfinite(ta):
                continue

            if ta > params.amplitude_threshold and params.relpos_lower <= rp <= params.relpos_upper:
                strength = (params.relpos_upper - rp) / band_width
                strength = float(np.clip(strength, 0.0, 1.0))
                this_size = floor + strength * size_range
                if not np.isfinite(this_size) or this_size <= 0.0:
                    this_size = floor
                signal[i] = 1
                size[i] = this_size
                active_size = this_size
                bars_remaining = max(hold - 1, 0)

        df = pd.DataFrame({"signal": signal.astype(np.int64), "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df.loc[~np.isfinite(df["size"]), "size"] = 1.0
        df.loc[df["size"] <= 0.0, "size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
