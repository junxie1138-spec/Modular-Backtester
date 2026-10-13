from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolumeShockParams:
    lookback: int = 20
    move_threshold: float = 1.6
    volume_threshold: float = 1.5
    hold_bars: int = 2
    cooldown_bars: int = 1


class GeneratedStrategy(BaseStrategy[VolumeShockParams]):
    strategy_id = "gen_local_1791297666"

    @classmethod
    def params_type(cls) -> type[VolumeShockParams]:
        return VolumeShockParams

    @classmethod
    def warmup_bars(cls, params: VolumeShockParams) -> int:
        return params.lookback + 2

    def indicators(self, data: pd.DataFrame, params: VolumeShockParams) -> pd.DataFrame:
        ret = data["close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        up = ret.gt(0)
        down = ret.lt(0)
        window = params.lookback
        move_base = ret.abs().shift(1).rolling(window, min_periods=window).mean()
        up_volume = data["volume"].where(up, 0.0).shift(1).rolling(window, min_periods=window).sum()
        down_volume = data["volume"].where(down, 0.0).shift(1).rolling(window, min_periods=window).sum()
        up_count = up.astype(float).shift(1).rolling(window, min_periods=window).sum()
        down_count = down.astype(float).shift(1).rolling(window, min_periods=window).sum()
        up_base = up_volume / up_count.replace(0, np.nan)
        down_base = down_volume / down_count.replace(0, np.nan)
        volume_base = pd.Series(np.where(up, up_base, down_base), index=data.index)
        result = pd.DataFrame(index=data.index)
        result["return"] = ret
        result["move_ratio"] = (ret.abs() / move_base.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result["volume_ratio"] = (data["volume"] / volume_base.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return result.fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VolumeShockParams,
    ) -> SignalFrame:
        qualified = (
            (indicators["move_ratio"] >= params.move_threshold)
            & (indicators["volume_ratio"] >= params.volume_threshold)
            & indicators["return"].ne(0)
        )
        strength = (
            (indicators["move_ratio"] / params.move_threshold - 1)
            + (indicators["volume_ratio"] / params.volume_threshold - 1)
        ).clip(0.0, 1.0)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.25, dtype=float)
        direction = 0
        exit_at = 0
        next_entry = 0
        entry_size = 0.25
        for i in range(len(data)):
            if direction and i >= exit_at:
                direction = 0
            if not direction and i >= next_entry and qualified.iloc[i]:
                direction = -int(np.sign(indicators["return"].iloc[i]))
                entry_size = float(0.25 + 0.70 * strength.iloc[i])
                exit_at = i + max(1, params.hold_bars)
                next_entry = exit_at + max(1, params.cooldown_bars)
            signal[i] = direction
            if direction:
                size[i] = entry_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
