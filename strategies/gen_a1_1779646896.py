from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    channel_window: int = 40
    accel_smoothing: int = 5
    percentile_window: int = 252
    percentile_threshold: float = 0.92
    hold_bars: int = 17
    trend_filter_window: int = 100
    rel_pos_ceiling: float = 0.85
    use_trend_filter: bool = True


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1779646896"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(
            params.channel_window
            + params.accel_smoothing
            + params.percentile_window
            + params.trend_filter_window
            + 5
        )

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        cw = int(max(2, params.channel_window))
        rolling_max = high.rolling(cw, min_periods=cw).max()
        rolling_min = low.rolling(cw, min_periods=cw).min()
        channel_width = (rolling_max - rolling_min).replace(0.0, np.nan)

        rel_pos = (close - rolling_min) / channel_width
        rel_pos = rel_pos.clip(0.0, 1.0)

        sm = int(max(2, params.accel_smoothing))
        velocity_raw = rel_pos.diff()
        velocity = velocity_raw.rolling(sm, min_periods=sm).mean()
        acceleration = velocity.diff()

        pw = int(max(20, params.percentile_window))
        min_p = max(20, pw // 4)
        accel_pct_rank = acceleration.rolling(pw, min_periods=min_p).rank(pct=True)

        tw = int(max(2, params.trend_filter_window))
        trend_ma = close.rolling(tw, min_periods=tw).mean()
        trend_up = (close > trend_ma).astype(float)

        out = pd.DataFrame(
            {
                "rel_pos": rel_pos,
                "velocity": velocity,
                "acceleration": acceleration,
                "accel_pct_rank": accel_pct_rank,
                "trend_up": trend_up,
            },
            index=data.index,
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)

        accel_rank = indicators["accel_pct_rank"].fillna(0.0).to_numpy()
        trend_up = indicators["trend_up"].fillna(0.0).to_numpy()
        rel_pos = indicators["rel_pos"].fillna(0.5).to_numpy()

        threshold = float(params.percentile_threshold)
        ceiling = float(params.rel_pos_ceiling)

        cond_now = accel_rank >= threshold
        cond_prev = np.zeros(n, dtype=bool)
        if n > 1:
            cond_prev[1:] = accel_rank[:-1] < threshold
        cross_up = cond_now & cond_prev & (rel_pos < ceiling)

        if params.use_trend_filter:
            cross_up = cross_up & (trend_up > 0.5)

        hold = int(max(1, params.hold_bars))
        raw_signal = np.zeros(n, dtype=np.int64)

        in_pos_until = -1
        for i in range(n):
            if cross_up[i] and i > in_pos_until:
                end = min(n, i + hold)
                raw_signal[i:end] = 1
                in_pos_until = end - 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = (
            pd.Series(raw_signal, index=data.index).shift(1).fillna(0).astype(int)
        )
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
