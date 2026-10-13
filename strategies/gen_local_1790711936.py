from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    breakout_lookback: int = 20
    volume_lookback: int = 20
    volume_multiple: float = 1.2
    season_start_day: int = 1
    season_end_day: int = 90
    atr_lookback: int = 14
    atr_multiple: float = 3.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790711936"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return max(params.breakout_lookback, params.volume_lookback, params.atr_lookback) + 1

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        prior_high = data["high"].rolling(params.breakout_lookback, min_periods=params.breakout_lookback).max().shift(1)
        avg_volume = data["volume"].rolling(params.volume_lookback, min_periods=params.volume_lookback).mean()
        prev_close = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prev_close).abs(),
            (data["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_lookback, min_periods=params.atr_lookback).mean()
        day = data.index.dayofyear
        if params.season_start_day <= params.season_end_day:
            in_season = (day >= params.season_start_day) & (day <= params.season_end_day)
        else:
            in_season = (day >= params.season_start_day) | (day <= params.season_end_day)
        return pd.DataFrame({
            "prior_high": prior_high,
            "avg_volume": avg_volume,
            "atr": atr,
            "in_season": in_season.astype(bool),
        }, index=data.index)

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        volume_confirmed = data["volume"] > indicators["avg_volume"] * params.volume_multiple
        breakout = data["close"] > indicators["prior_high"]
        eligible = indicators["in_season"] & volume_confirmed & breakout
        confirmed = eligible & eligible.shift(1, fill_value=False)

        raw = pd.Series(0, index=data.index, dtype=int)
        in_trade = False
        high_water = 0.0
        for i in range(len(data)):
            close = float(data["close"].iloc[i])
            atr = indicators["atr"].iloc[i]
            if not in_trade:
                if bool(confirmed.iloc[i]) and np.isfinite(atr):
                    raw.iloc[i] = 1
                    in_trade = True
                    high_water = close
                continue
            high_water = max(high_water, close)
            if np.isfinite(atr) and close <= high_water - params.atr_multiple * float(atr):
                raw.iloc[i] = 0
                in_trade = False
                high_water = 0.0
            else:
                raw.iloc[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw.shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
