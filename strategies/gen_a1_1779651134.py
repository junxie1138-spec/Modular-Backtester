from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveReleaseParams:
    jam_window: int = 20
    jam_lookback: int = 252
    jam_percentile: float = 0.80
    flux_window: int = 5
    holding_bars: int = 7


class GeneratedStrategy(BaseStrategy[ShockwaveReleaseParams]):
    strategy_id = "gen_a1_1779651134"

    @classmethod
    def params_type(cls) -> type[ShockwaveReleaseParams]:
        return ShockwaveReleaseParams

    def warmup_bars(self, params: ShockwaveReleaseParams) -> int:
        return int(params.jam_lookback + params.jam_window + max(2 * params.flux_window, 2))

    def indicators(self, data: pd.DataFrame, params: ShockwaveReleaseParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        ret = close.pct_change()

        neg_sq = ret.where(ret < 0, 0.0).pow(2)
        density = neg_sq.rolling(params.jam_window, min_periods=params.jam_window).sum()

        density_rank = density.rolling(
            params.jam_lookback, min_periods=params.jam_lookback
        ).rank(pct=True)

        flux_now = ret.rolling(params.flux_window, min_periods=params.flux_window).sum()
        flux_prev = flux_now.shift(params.flux_window)
        flux_accel = flux_now - flux_prev

        out = pd.DataFrame(index=data.index)
        out["ret"] = ret
        out["density"] = density
        out["density_rank"] = density_rank
        out["flux_now"] = flux_now
        out["flux_accel"] = flux_accel
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockwaveReleaseParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)

        jam_critical = indicators["density_rank"] >= float(params.jam_percentile)
        flux_positive = indicators["flux_now"] > 0.0
        flux_accelerating = indicators["flux_accel"] > 0.0

        entry = (jam_critical & flux_positive & flux_accelerating).fillna(False).to_numpy()

        n = len(df)
        raw = np.zeros(n, dtype=np.int64)
        holding = max(int(params.holding_bars), 1)
        bars_remaining = 0
        for i in range(n):
            if bars_remaining > 0:
                raw[i] = 1
                bars_remaining -= 1
            elif entry[i]:
                raw[i] = 1
                bars_remaining = holding - 1

        df["signal_raw"] = raw
        df["size"] = 1.0
        df["signal"] = df["signal_raw"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
