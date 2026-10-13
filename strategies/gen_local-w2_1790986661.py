from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationParams:
    scale_window: int = 20
    force_threshold: float = 0.65
    sizing_gain: float = 0.30


class GeneratedStrategy(BaseStrategy[AccelerationParams]):
    strategy_id = "gen_local-w2_1790986661"

    @classmethod
    def params_type(cls) -> type[AccelerationParams]:
        return AccelerationParams

    def warmup_bars(self, params: AccelerationParams) -> int:
        return max(2, int(params.scale_window)) + 3

    def indicators(self, data: pd.DataFrame, params: AccelerationParams) -> pd.DataFrame:
        window = max(2, int(params.scale_window))
        close = data["close"].astype(float)
        opening = data["open"].astype(float)
        previous = close.shift(1).where(close.shift(1) > 0.0)
        roc = close.div(previous).sub(1.0)
        acceleration = roc.diff()
        scale = acceleration.abs().shift(1).rolling(window, min_periods=window).mean()
        candle_range = (data["high"] - data["low"]).where(lambda x: x > 0.0)
        location = ((2.0 * close - data["high"] - data["low"]) / candle_range).clip(-1.0, 1.0)
        body = close - opening
        gap = opening - close.shift(1)
        total_movement = body.abs() + gap.abs()
        intraday_share = body.abs().div(total_movement.where(total_movement > 0.0))
        force = acceleration.div(scale.where(scale > 1e-12)) * intraday_share * location.abs()
        finite = np.isfinite(force) & np.isfinite(roc) & np.isfinite(location)
        aligned = (acceleration * roc > 0.0) & (body * roc > 0.0) & (location * roc > 0.0)
        out = pd.DataFrame(index=data.index)
        out["roc"] = roc
        out["acceleration"] = acceleration
        out["acceleration_scale"] = scale
        out["closing_location"] = location
        out["intraday_share"] = intraday_share
        out["force"] = force
        out["eligible"] = (finite & aligned & scale.notna()).astype(int)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AccelerationParams,
    ) -> SignalFrame:
        threshold = max(0.01, float(params.force_threshold))
        force = indicators["force"].to_numpy(dtype=float)
        eligible = indicators["eligible"].to_numpy(dtype=bool)
        condition = np.where(eligible & (np.abs(force) >= threshold), -np.sign(force), 0).astype(int)
        entry_sizes = np.clip(0.20 + max(0.0, float(params.sizing_gain)) * np.abs(force) / threshold, 0.20, 0.95)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.20, dtype=float)
        position = 0
        position_size = 0.20
        for i, candidate in enumerate(condition):
            if position == 0:
                if candidate != 0:
                    position = int(candidate)
                    position_size = float(entry_sizes[i])
            elif candidate == -position:
                # Only an opposite qualifying entry condition closes a trade.
                position = 0
                position_size = 0.20
            signals[i] = position
            sizes[i] = position_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
