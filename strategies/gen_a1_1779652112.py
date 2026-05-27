from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapSpringTensionParams:
    ma_window: int = 200
    tension_window: int = 10
    compression_max_pct: float = 0.003
    yield_pct: float = 0.015
    tension_threshold: float = 0.008
    min_trigger_gap_pct: float = 0.0005
    profit_target_pct: float = 0.015
    max_hold_bars: int = 2
    base_size: float = 1.0
    tension_size_scale: float = 50.0


class GeneratedStrategy(BaseStrategy[GapSpringTensionParams]):
    strategy_id = "gen_a1_1779652112"

    @classmethod
    def params_type(cls):
        return GapSpringTensionParams

    @classmethod
    def warmup_bars(cls, params):
        return int(max(params.ma_window, params.tension_window) + 2)

    def indicators(self, data, params):
        close = data["close"]
        open_ = data["open"]
        prev_close = close.shift(1)

        gap_pct = (open_ - prev_close) / prev_close

        is_compression = (gap_pct < 0) & (gap_pct.abs() <= params.compression_max_pct)
        compression = (-gap_pct).where(is_compression, 0.0).fillna(0.0)

        yield_event = (gap_pct.abs() >= params.yield_pct).fillna(False)

        compression_cum = compression.cumsum()
        yield_cum = compression_cum.where(yield_event).ffill().fillna(0.0)
        since_yield = compression_cum - yield_cum

        rolling_compression = compression.rolling(params.tension_window, min_periods=1).sum()
        tension = pd.concat([rolling_compression, since_yield], axis=1).min(axis=1).fillna(0.0)

        ma = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        regime_on = (close > ma).fillna(False)

        release = (gap_pct >= params.min_trigger_gap_pct) & (tension >= params.tension_threshold) & regime_on
        release = release.fillna(False)

        return pd.DataFrame({
            "gap_pct": gap_pct,
            "tension": tension,
            "regime_on": regime_on.astype(float),
            "release": release.astype(float),
            "ma": ma,
        })

    def generate_signals(self, data, indicators, ctx, params):
        n = len(data)
        raw_signal = np.zeros(n, dtype=np.int64)
        raw_size = np.full(n, params.base_size, dtype=np.float64)

        close = data["close"].to_numpy()
        release = indicators["release"].to_numpy()
        tension = indicators["tension"].to_numpy()

        in_pos = False
        entry_close = 0.0
        bars_in = 0

        for i in range(n):
            if in_pos:
                bars_in += 1
                gain = (close[i] - entry_close) / entry_close if entry_close > 0 else 0.0
                if gain >= params.profit_target_pct or bars_in >= params.max_hold_bars:
                    in_pos = False
                    entry_close = 0.0
                    bars_in = 0
                    raw_signal[i] = 0
                else:
                    raw_signal[i] = 1
                    t = tension[i] if not np.isnan(tension[i]) else 0.0
                    raw_size[i] = params.base_size + params.tension_size_scale * t
            else:
                if release[i] >= 1.0 and not np.isnan(close[i]):
                    in_pos = True
                    entry_close = close[i]
                    bars_in = 0
                    raw_signal[i] = 1
                    t = tension[i] if not np.isnan(tension[i]) else 0.0
                    raw_size[i] = params.base_size + params.tension_size_scale * t
                else:
                    raw_signal[i] = 0

        df = pd.DataFrame({"signal": raw_signal, "size": raw_size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.base_size)
        df["size"] = df["size"].clip(lower=1e-6).astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
