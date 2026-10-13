from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolumeResumptionParams:
    volume_window: int = 20
    flow_window: int = 12
    flow_threshold: float = 4.0
    trigger_volume: float = 1.5
    impulse_mult: float = 1.2
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[VolumeResumptionParams]):
    strategy_id = "gen_local_1791223864"

    @classmethod
    def params_type(cls) -> type[VolumeResumptionParams]:
        return VolumeResumptionParams

    @classmethod
    def warmup_bars(cls, params: VolumeResumptionParams) -> int:
        return params.volume_window + params.flow_window + 4

    def indicators(self, data: pd.DataFrame, params: VolumeResumptionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change().fillna(0.0)
        median_volume = data["volume"].rolling(params.volume_window, min_periods=params.volume_window).median()
        out["relative_volume"] = data["volume"].div(median_volume.replace(0, np.nan)).fillna(0.0)
        weighted_direction = np.sign(out["return"]) * out["relative_volume"].clip(upper=3.0)
        out["flow"] = weighted_direction.rolling(params.flow_window, min_periods=params.flow_window).sum().shift(3).fillna(0.0)
        out["typical_move"] = out["return"].abs().rolling(params.volume_window, min_periods=params.volume_window).median().fillna(0.0)
        quiet = (out["relative_volume"].shift(1) < 1.0) & (out["relative_volume"].shift(2) < 1.0)
        out["pullback_down"] = quiet & (out["return"].shift(1) < 0) & (out["return"].shift(2) < 0)
        out["pullback_up"] = quiet & (out["return"].shift(1) > 0) & (out["return"].shift(2) > 0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VolumeResumptionParams,
    ) -> SignalFrame:
        move = indicators["return"]
        volume = indicators["relative_volume"]
        flow = indicators["flow"]
        enough_volume = volume >= params.trigger_volume
        long_entry = (
            (flow >= params.flow_threshold)
            & indicators["pullback_down"]
            & enough_volume
            & (move >= params.impulse_mult * indicators["typical_move"])
        )
        short_entry = (
            (flow <= -params.flow_threshold)
            & indicators["pullback_up"]
            & enough_volume
            & (move <= -params.impulse_mult * indicators["typical_move"])
        )
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        conviction = (0.25 + 0.35 * (flow.abs() / params.flow_window).clip(upper=1.0)
                      + 0.35 * (volume / (2 * params.trigger_volume)).clip(upper=1.0)).clip(upper=0.95).to_numpy()

        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.5, dtype=float)
        position = 0
        exit_at = -1
        active_size = 0.5
        for i in range(len(data)):
            if position and i == exit_at:
                position = 0
                continue
            if not position and entries[i]:
                position = int(entries[i])
                exit_at = i + params.hold_bars
                active_size = float(conviction[i])
            signals[i] = position
            if position:
                sizes[i] = active_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.5).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
