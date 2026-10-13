from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    breakout_lookback: int = 20
    volume_window: int = 20
    volume_z_min: float = 0.5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790712182"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return max(params.breakout_lookback, params.volume_window) + 1

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        prior_high = data["high"].shift(1).rolling(params.breakout_lookback).max()
        prior_low = data["low"].shift(1).rolling(params.breakout_lookback).min()
        volume_mean = data["volume"].rolling(params.volume_window).mean()
        volume_std = data["volume"].rolling(params.volume_window).std(ddof=0).replace(0, np.nan)
        volume_z = (data["volume"] - volume_mean) / volume_std
        return pd.DataFrame(
            {
                "prior_high": prior_high,
                "prior_low": prior_low,
                "volume_z": volume_z,
            },
            index=data.index,
        )

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        valid = indicators[["prior_high", "prior_low", "volume_z"]].notna().all(axis=1)
        active_volume = indicators["volume_z"].ge(params.volume_z_min)
        long_bar = valid & active_volume & data["close"].gt(indicators["prior_high"])
        short_bar = valid & active_volume & data["close"].lt(indicators["prior_low"])
        long_entry = long_bar & long_bar.shift(1, fill_value=False)
        short_entry = short_bar & short_bar.shift(1, fill_value=False)

        signal = np.zeros(len(data), dtype=int)
        position = 0
        for i, (go_long, go_short) in enumerate(zip(long_entry.to_numpy(), short_entry.to_numpy())):
            if position == 1 and go_short:
                position = 0
            elif position == -1 and go_long:
                position = 0
            elif position == 0:
                if go_long:
                    position = 1
                elif go_short:
                    position = -1
            signal[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
