from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TrendDutyParams:
    ma_len: int = 20
    dist_std_len: int = 20
    duty_len: int = 15
    entry_threshold: float = 0.15
    profit_target: float = 0.02
    time_stop: int = 2
    base_size: float = 1.0
    size_scale: float = 1.5
    max_size_mult: float = 2.5


class GeneratedStrategy(BaseStrategy[TrendDutyParams]):
    """Trend-strength via the sign duty-cycle of the distance-from-MA z-score.

    The moving average is the node of a standing wave; the distance-from-MA
    z-score is the displacement. trend_strength = (fraction of recent bars with
    positive z-score) - 0.5 measures which antinode side the wave is camped on.
    A cross of that occupancy past a skew threshold marks a freshly established
    trend; size scales with how far past the threshold the occupancy sits.
    """

    strategy_id = "gen_a1_1778898235"

    @classmethod
    def params_type(cls):
        return TrendDutyParams

    @staticmethod
    def warmup_bars(params: TrendDutyParams) -> int:
        return int(params.ma_len + params.dist_std_len + params.duty_len + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: TrendDutyParams) -> pd.DataFrame:
        close = data["close"].astype(float)

        ma = close.rolling(params.ma_len).mean()
        distance = close - ma

        dist_std = distance.rolling(params.dist_std_len).std()
        dist_std = dist_std.where(dist_std > 0.0)
        z = distance / dist_std

        # 1.0 where price is above its MA, 0.0 where below, NaN while z undefined.
        up = pd.Series(np.where(z.to_numpy() > 0.0, 1.0, 0.0), index=close.index)
        up = up.where(z.notna())

        # rolling mean uses min_periods == window, so duty is NaN until a full
        # window of valid z-scores exists - no warmup contamination.
        duty = up.rolling(params.duty_len).mean()
        trend_strength = duty - 0.5

        out = pd.DataFrame(index=data.index)
        out["ma"] = ma
        out["z"] = z
        out["duty"] = duty
        out["trend_strength"] = trend_strength
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TrendDutyParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        ts = indicators["trend_strength"].to_numpy(dtype=float)
        n = len(close)

        signal = np.zeros(n, dtype=int)
        size = np.full(n, float(params.base_size), dtype=float)

        thr = float(params.entry_threshold)
        span = max(0.5 - thr, 1e-9)

        cur = 0
        entry_idx = -1
        entry_price = 0.0
        conv = 1.0

        for i in range(n):
            if cur == 0:
                if i > 0 and np.isfinite(ts[i]) and np.isfinite(ts[i - 1]):
                    long_cross = ts[i - 1] <= thr and ts[i] > thr
                    short_cross = ts[i - 1] >= -thr and ts[i] < -thr
                    if long_cross:
                        cur = 1
                    elif short_cross:
                        cur = -1
                    if cur != 0:
                        entry_idx = i
                        entry_price = close[i]
                        excess = (abs(ts[i]) - thr) / span
                        if excess < 0.0:
                            excess = 0.0
                        if excess > 1.0:
                            excess = 1.0
                        conv = 1.0 + params.size_scale * excess
                        if conv > params.max_size_mult:
                            conv = params.max_size_mult
                        if conv < 1.0:
                            conv = 1.0
                signal[i] = cur
                size[i] = params.base_size * conv if cur != 0 else params.base_size
            else:
                held = i - entry_idx
                if entry_price > 0.0:
                    ret = (close[i] - entry_price) / entry_price * cur
                else:
                    ret = 0.0
                if ret >= params.profit_target or held >= params.time_stop:
                    cur = 0
                    entry_idx = -1
                    signal[i] = 0
                    size[i] = params.base_size
                else:
                    signal[i] = cur
                    size[i] = params.base_size * conv

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal

        size_series = pd.Series(size, index=data.index)
        # keep conviction-scaled size paired with the bar it decided on
        size_series = size_series.shift(1).fillna(float(params.base_size))
        size_series = size_series.where(size_series > 0.0, float(params.base_size))
        df["size"] = size_series.astype(float)

        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
