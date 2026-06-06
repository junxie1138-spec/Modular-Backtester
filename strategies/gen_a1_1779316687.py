from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionShockwaveParams:
    compression_lookback: int = 20
    compression_threshold: float = 0.72
    upper_close_min: float = 0.68
    min_pressure_streak: int = 4
    atr_period: int = 14
    atr_stop_k: float = 1.6
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[CompressionShockwaveParams]):
    strategy_id = "gen_a1_1779316687"

    @classmethod
    def params_type(cls):
        return CompressionShockwaveParams

    @classmethod
    def warmup_bars(cls, params: CompressionShockwaveParams) -> int:
        return int(max(params.compression_lookback, params.atr_period) + 2)

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: CompressionShockwaveParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        open_ = data["open"]

        hl_range = high - low
        range_safe = hl_range.replace(0.0, np.nan)

        close_loc = ((close - low) / range_safe).clip(0.0, 1.0)
        rolling_range_median = hl_range.rolling(params.compression_lookback, min_periods=params.compression_lookback).median()
        compression_ratio = hl_range / rolling_range_median.replace(0.0, np.nan)

        compressed_pressure = (
            (compression_ratio <= params.compression_threshold)
            & (close_loc >= params.upper_close_min)
            & (close >= open_)
        ).fillna(False)

        pressure_streak = compressed_pressure.astype(int).groupby((~compressed_pressure).cumsum()).cumcount() + 1
        pressure_streak = pressure_streak.where(compressed_pressure, 0).astype(int)

        prev_high = high.shift(1)
        prev_range = hl_range.shift(1)
        expansion_up = ((close > prev_high) & (hl_range > prev_range) & (close > open_)).fillna(False)

        entry_raw = (
            (pressure_streak.shift(2) >= params.min_pressure_streak)
            & expansion_up.shift(1)
            & expansion_up
        ).fillna(False)

        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                hl_range,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_period, min_periods=params.atr_period).mean()

        return pd.DataFrame(
            {
                "compression_ratio": compression_ratio,
                "close_loc": close_loc,
                "pressure_streak": pressure_streak,
                "expansion_up": expansion_up.astype(int),
                "entry_raw": entry_raw.astype(int),
                "atr": atr,
            },
            index=data.index,
        )

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CompressionShockwaveParams,
    ) -> SignalFrame:
        close = data["close"]
        atr = indicators["atr"]
        entry_raw = indicators["entry_raw"].fillna(0).astype(int)

        signal = np.zeros(len(data), dtype=int)
        size = np.ones(len(data), dtype=float)

        in_position = False
        entry_price = 0.0
        stop_level = np.nan
        bars_in_trade = 0

        for i in range(len(data)):
            current_close = float(close.iat[i])
            current_atr = float(atr.iat[i]) if pd.notna(atr.iat[i]) else np.nan

            if not in_position:
                if entry_raw.iat[i] == 1 and pd.notna(current_atr) and current_atr > 0.0:
                    in_position = True
                    entry_price = current_close
                    stop_level = entry_price - params.atr_stop_k * current_atr
                    bars_in_trade = 0
                    signal[i] = 1
                else:
                    signal[i] = 0
            else:
                bars_in_trade += 1
                stop_hit = current_close < stop_level
                time_exit = bars_in_trade >= params.max_hold_bars

                if stop_hit or time_exit:
                    in_position = False
                    signal[i] = 0
                    entry_price = 0.0
                    stop_level = np.nan
                    bars_in_trade = 0
                else:
                    signal[i] = 1

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
