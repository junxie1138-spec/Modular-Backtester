from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveEscapeParams:
    entry_z: float = 1.2
    trail_atr: float = 1.2


class GeneratedStrategy(BaseStrategy[WaveEscapeParams]):
    strategy_id = "gen_local_1790992622"

    @classmethod
    def params_type(cls) -> type[WaveEscapeParams]:
        return WaveEscapeParams

    def warmup_bars(self, params: WaveEscapeParams) -> int:
        return 26

    def indicators(self, data: pd.DataFrame, params: WaveEscapeParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        mean = close.rolling(20, min_periods=20).mean()
        std = close.rolling(20, min_periods=20).std(ddof=0).replace(0.0, np.nan)
        z = ((close - mean) / std).replace([np.inf, -np.inf], np.nan)
        previous_z = z.shift(1)
        crossings = (z * previous_z < 0.0).astype(float)
        crossings = crossings.where(z.notna() & previous_z.notna())
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        atr = true_range.rolling(14, min_periods=14).mean()
        return pd.DataFrame({
            "mean": mean,
            "std": std,
            "z": z,
            "crossing": crossings,
            "prior_band": z.abs().shift(1).rolling(5, min_periods=5).max(),
            "prior_crossings": crossings.shift(1).rolling(5, min_periods=5).sum(),
            "true_range": true_range,
            "atr": atr,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveEscapeParams,
    ) -> SignalFrame:
        if params.entry_z <= 0.0 or params.trail_atr <= 0.0:
            raise ValueError("Parameters must be positive")
        z = indicators["z"]
        ready = (
            indicators[["z", "prior_band", "prior_crossings", "atr"]]
            .replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
            & indicators["atr"].gt(0.0)
        )
        balanced = (
            ready
            & indicators["prior_band"].lt(params.entry_z)
            & indicators["prior_crossings"].ge(2.0)
        )
        entries = np.select([
            balanced & z.ge(params.entry_z),
            balanced & z.le(-params.entry_z),
        ], [1, -1], default=0).astype(int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        zs = z.to_numpy(dtype=float)
        valid = ready.to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = low_water = stop = np.nan
        for i in range(len(data)):
            if not valid[i] or not np.isfinite(closes[i]):
                position = 0
                age = 0
                high_water = low_water = stop = np.nan
                continue
            if position:
                age += 1
                # First held session follows the entry decision by one bar.
                if position == 1:
                    high_water = closes[i] if age == 1 else max(high_water, closes[i])
                    candidate = high_water - params.trail_atr * atrs[i]
                    stop = candidate if age == 1 else max(stop, candidate)
                    stopped = closes[i] <= stop
                else:
                    low_water = closes[i] if age == 1 else min(low_water, closes[i])
                    candidate = low_water + params.trail_atr * atrs[i]
                    stop = candidate if age == 1 else min(stop, candidate)
                    stopped = closes[i] >= stop
                if stopped or age >= 2 or position * zs[i] <= 0.5 * params.entry_z:
                    position = 0
                    age = 0
                    high_water = low_water = stop = np.nan
            elif entries[i] != 0:
                position = int(entries[i])
                age = 0
                high_water = low_water = stop = np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
