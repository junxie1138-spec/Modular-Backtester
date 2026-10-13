from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RankLeadParams:
    rank_window: int = 24
    atr_multiple: float = 1.5


class GeneratedStrategy(BaseStrategy[RankLeadParams]):
    strategy_id = "gen_local_1790996749"

    @classmethod
    def params_type(cls) -> type[RankLeadParams]:
        return RankLeadParams

    def warmup_bars(self, params: RankLeadParams) -> int:
        return max(max(2, int(params.rank_window)) + 1, 15)

    def indicators(self, data: pd.DataFrame, params: RankLeadParams) -> pd.DataFrame:
        n = max(2, int(params.rank_window))
        close = data["close"].astype(float)
        midpoint = (data["high"] + data["low"]) / 2.0
        previous_close = close.shift(1)
        tr = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["midpoint"] = midpoint
        out["true_range"] = tr
        out["close_rank"] = close.rolling(n, min_periods=n).rank(pct=True)
        out["midpoint_rank"] = midpoint.rolling(n, min_periods=n).rank(pct=True)
        out["rank_lead"] = out["close_rank"] - out["midpoint_rank"]
        out["previous_rank_lead"] = out["rank_lead"].shift(1)
        out["atr"] = tr.rolling(14, min_periods=14).mean()
        span = (data["high"] - data["low"]).replace(0.0, np.nan)
        out["body_fraction"] = (close - data["open"]) / span
        ready = np.isfinite(out).all(axis=1) & (out["atr"] > 0.0)
        long_entry = (
            (out["rank_lead"] >= 0.15)
            & (out["previous_rank_lead"] < 0.15)
            & (out["body_fraction"] >= 0.5)
        )
        short_entry = (
            (out["rank_lead"] <= -0.15)
            & (out["previous_rank_lead"] > -0.15)
            & (out["body_fraction"] <= -0.5)
        )
        out["entry"] = np.select(
            [ready & long_entry, ready & short_entry], [1, -1], default=0
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RankLeadParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = low_water = high_stop = short_stop = 0.0
        k = max(0.1, float(params.atr_multiple))
        for i in range(len(data)):
            if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0.0:
                position = 0
                age = 0
                continue
            if position != 0:
                age += 1
                high_water = max(high_water, close[i])
                low_water = min(low_water, close[i])
                # Highest-close stop applies to both directions and only rises.
                high_stop = max(high_stop, high_water - k * atr[i])
                # Shorts also have a mirrored adverse-move protection.
                if position == -1:
                    short_stop = min(short_stop, low_water + k * atr[i])
                exit_now = (
                    close[i] <= high_stop
                    or (position == -1 and close[i] >= short_stop)
                    or age >= 4
                )
                if exit_now:
                    position = 0
                    age = 0
                    continue
            elif entries[i] != 0:
                position = int(entries[i])
                age = 0
                high_water = low_water = close[i]
                high_stop = close[i] - k * atr[i]
                short_stop = close[i] + k * atr[i]
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
