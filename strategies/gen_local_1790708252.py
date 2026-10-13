from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PathAreaParams:
    area_bars: int = 12
    volume_window: int = 20
    baseline_bars: int = 63
    trigger_z: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[PathAreaParams]):
    strategy_id = "gen_local_1790708252"

    @classmethod
    def params_type(cls):
        return PathAreaParams

    @classmethod
    def warmup_bars(cls, params: PathAreaParams) -> int:
        return int(params.volume_window + params.area_bars + params.baseline_bars + 1)

    def indicators(self, data: pd.DataFrame, params: PathAreaParams) -> pd.DataFrame:
        x = np.log(data["close"].clip(lower=1e-12)).ffill().fillna(0.0)
        volume_mean = data["volume"].rolling(params.volume_window, min_periods=params.volume_window).mean()
        activity = (data["volume"] / volume_mean.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = activity.cumsum()
        dx = x.diff().fillna(0.0)
        cross = (x.shift(1) * activity - y.shift(1) * dx).fillna(0.0)
        x0 = x.shift(params.area_bars)
        y0 = y.shift(params.area_bars)
        area = 0.5 * (cross.rolling(params.area_bars, min_periods=params.area_bars).sum() - x0 * (y - y0) + y0 * (x - x0))
        volatility = dx.rolling(params.area_bars, min_periods=params.area_bars).std()
        scale = (y - y0) * volatility * np.sqrt(params.area_bars)
        normalized = (area / scale.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        baseline = normalized.rolling(params.baseline_bars, min_periods=params.baseline_bars)
        area_z = ((normalized - baseline.mean()) / baseline.std().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({"signature_area": area.fillna(0.0), "area_z": area_z, "last_return": dx}, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: PathAreaParams) -> SignalFrame:
        z = indicators["area_z"]
        last_return = indicators["last_return"]
        long_entry = (z >= params.trigger_z) & (last_return < 0)
        short_entry = (z <= -params.trigger_z) & (last_return > 0)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        position = 0
        entry_bar = -1
        position_size = 0.25

        for i in range(len(data)):
            if position and i - entry_bar >= params.hold_bars:
                position = 0
                position_size = 0.25
            elif position == 0 and (long_entry.iloc[i] or short_entry.iloc[i]):
                position = 1 if long_entry.iloc[i] else -1
                entry_bar = i
                strength = abs(float(z.iloc[i])) - params.trigger_z
                position_size = float(np.clip(0.25 + 0.35 * strength, 0.25, 0.95))
            signals[i] = position
            sizes[i] = position_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
