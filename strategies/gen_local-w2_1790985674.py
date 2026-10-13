from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DiffusionParams:
    roc_bars: int = 5
    acceleration_bars: int = 3
    diffusion_bars: int = 20
    volatility_bars: int = 20
    acceleration_threshold: float = 0.05
    occupancy_floor: float = 0.45
    occupancy_ceiling: float = 0.75
    annual_vol_target: float = 0.12
    holding_bars: int = 8
    min_size: float = 0.05
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[DiffusionParams]):
    strategy_id = "gen_local-w2_1790985674"

    @classmethod
    def params_type(cls) -> type[DiffusionParams]:
        return DiffusionParams

    def warmup_bars(self, params: DiffusionParams) -> int:
        return int(max(
            params.roc_bars + params.acceleration_bars + 1,
            params.diffusion_bars + 1,
            params.volatility_bars + 1,
        ) + 1)

    def indicators(self, data: pd.DataFrame, params: DiffusionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        out["return"] = close.div(close.shift(1)).sub(1)
        out["roc"] = close.div(close.shift(params.roc_bars)).sub(1)
        out["acceleration"] = out["roc"].diff(params.acceleration_bars).div(params.acceleration_bars)
        out["daily_vol"] = out["return"].rolling(
            params.volatility_bars, min_periods=params.volatility_bars
        ).std(ddof=0)
        valid_return = out["return"].notna()
        out["up_occupancy"] = out["return"].gt(0).astype(float).where(valid_return).rolling(
            params.diffusion_bars, min_periods=params.diffusion_bars
        ).mean()
        out["down_occupancy"] = out["return"].lt(0).astype(float).where(valid_return).rolling(
            params.diffusion_bars, min_periods=params.diffusion_bars
        ).mean()
        out["roc_scale"] = out["daily_vol"].where(out["daily_vol"] > 0) * np.sqrt(params.roc_bars)
        # Remaining directional participation discounts acceleration near saturation.
        out["long_pressure"] = out["acceleration"] * (1 - out["up_occupancy"]) / out["roc_scale"]
        out["short_pressure"] = -out["acceleration"] * (1 - out["down_occupancy"]) / out["roc_scale"]
        out["target_size"] = (
            params.annual_vol_target / (out["daily_vol"].where(out["daily_vol"] > 0) * np.sqrt(252))
        ).clip(params.min_size, params.max_size)
        out = out.replace([np.inf, -np.inf], np.nan)
        long_setup = (
            out["roc"].gt(0)
            & out["long_pressure"].gt(params.acceleration_threshold)
            & out["up_occupancy"].between(params.occupancy_floor, params.occupancy_ceiling)
            & out["target_size"].notna()
        )
        short_setup = (
            out["roc"].lt(0)
            & out["short_pressure"].gt(params.acceleration_threshold)
            & out["down_occupancy"].between(params.occupancy_floor, params.occupancy_ceiling)
            & out["target_size"].notna()
        )
        out["setup"] = np.select([long_setup, short_setup], [1, -1], default=0).astype(int)
        out["confirmed"] = out["setup"].where(
            out["setup"].ne(0) & out["setup"].eq(out["setup"].shift(1)), 0
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DiffusionParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        confirmed = indicators["confirmed"].to_numpy(dtype=int)
        target_size = indicators["target_size"].to_numpy(dtype=float)
        direction = 0
        entry_bar = -1
        entry_size = params.min_size
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            if direction != 0:
                if i - entry_bar >= params.holding_bars:
                    # Flat on expiry; entry checks resume on the following bar.
                    direction = 0
                    entry_bar = -1
                    entry_size = params.min_size
                else:
                    signals[i] = direction
                    sizes[i] = entry_size
                continue
            if i >= warmup - 1 and confirmed[i] != 0 and np.isfinite(target_size[i]):
                direction = int(confirmed[i])
                entry_bar = i
                entry_size = float(target_size[i])
                signals[i] = direction
                sizes[i] = entry_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
