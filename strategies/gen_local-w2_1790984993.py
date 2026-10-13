from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AuctionSlopeParams:
    compression_ratio: float = 0.75
    trail_atr: float = 1.6


class GeneratedStrategy(BaseStrategy[AuctionSlopeParams]):
    strategy_id = "gen_local-w2_1790984993"

    @classmethod
    def params_type(cls) -> type[AuctionSlopeParams]:
        return AuctionSlopeParams

    def warmup_bars(self, params: AuctionSlopeParams) -> int:
        return 28

    def indicators(self, data: pd.DataFrame, params: AuctionSlopeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        previous_close = close.shift(1)
        true_range = pd.concat([
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(14, min_periods=14).mean()
        baseline = true_range.shift(4).rolling(24, min_periods=24).mean()
        out["compression"] = true_range.rolling(4, min_periods=4).mean().div(
            baseline.where(baseline > 0)
        )
        midpoint = (high + low) / 2.0
        midpoint_slope = pd.Series(0.0, index=data.index)
        close_slope = pd.Series(0.0, index=data.index)
        # Six-point OLS slopes: oldest weight -2.5, newest weight +2.5.
        for lag in range(6):
            weight = 2.5 - lag
            midpoint_slope = midpoint_slope + weight * midpoint.shift(lag)
            close_slope = close_slope + weight * close.shift(lag)
        scale = out["atr"].where(out["atr"] > 0)
        out["midpoint_slope"] = (midpoint_slope / 17.5).div(scale)
        out["close_slope"] = (close_slope / 17.5).div(scale)
        out["slope_innovation"] = out["midpoint_slope"].diff()
        out["previous_midpoint_slope"] = out["midpoint_slope"].shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AuctionSlopeParams,
    ) -> SignalFrame:
        valid = indicators.notna().all(axis=1)
        entries = (
            valid
            & (indicators["compression"] < params.compression_ratio)
            & (indicators["previous_midpoint_slope"] <= 0.0)
            & (indicators["midpoint_slope"] > 0.0)
            & (indicators["slope_innovation"] > 0.05)
            & (indicators["close_slope"] <= 0.0)
            & (indicators["close_slope"] >= -0.10)
        ).fillna(False).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=np.int64)
        active = False
        age = 0
        high_water = -np.inf
        trailing_stop = -np.inf
        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            usable = np.isfinite(close) and np.isfinite(atr) and atr > 0.0
            if active:
                age += 1
                if usable:
                    # These closes occur after the next-bar entry fill.
                    high_water = max(high_water, close)
                    trailing_stop = max(
                        trailing_stop, high_water - params.trail_atr * atr
                    )
                if not usable or close <= trailing_stop or age >= 4:
                    active = False
                    age = 0
                    high_water = -np.inf
                    trailing_stop = -np.inf
                else:
                    signals[i] = 1
            elif entries[i] and usable:
                active = True
                age = 0
                high_water = -np.inf
                trailing_stop = -np.inf
                signals[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
