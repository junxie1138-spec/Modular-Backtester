from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BasinParams:
    excursion_window: int = 20
    atr_window: int = 14
    depth_atr: float = 1.5
    occupancy_window: int = 5
    occupancy_fraction: float = 0.8
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[BasinParams]):
    strategy_id = "gen_local-w2_1791233534"

    @classmethod
    def params_type(cls) -> type[BasinParams]:
        return BasinParams

    def warmup_bars(self, params: BasinParams) -> int:
        return int(max(200, params.excursion_window + params.occupancy_window - 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: BasinParams) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        close = data["close"]
        result["previous_close"] = close.shift(1)
        result["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - result["previous_close"]).abs(),
            (data["low"] - result["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        result["atr"] = result["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean().replace(0.0, np.nan)
        result["ma200"] = close.rolling(200, min_periods=200).mean()
        result["peak"] = close.rolling(params.excursion_window, min_periods=params.excursion_window).max()
        result["trough"] = close.rolling(params.excursion_window, min_periods=params.excursion_window).min()
        result["drawdown"] = result["peak"] - close
        result["rebound"] = close - result["trough"]
        result["down_depth"] = result["drawdown"] / result["atr"]
        result["up_depth"] = result["rebound"] / result["atr"]
        result["down_occupied"] = result["down_depth"].ge(params.depth_atr).astype(float).where(result["down_depth"].notna())
        result["up_occupied"] = result["up_depth"].ge(params.depth_atr).astype(float).where(result["up_depth"].notna())
        result["down_occupancy"] = result["down_occupied"].rolling(params.occupancy_window, min_periods=params.occupancy_window).mean()
        result["up_occupancy"] = result["up_occupied"].rolling(params.occupancy_window, min_periods=params.occupancy_window).mean()
        result["down_repair"] = result["drawdown"].diff().lt(0) & close.gt(result["previous_close"])
        result["up_repair"] = result["rebound"].diff().lt(0) & close.lt(result["previous_close"])
        result["long_entry"] = (
            close.gt(result["ma200"])
            & result["down_depth"].ge(params.depth_atr)
            & result["down_occupancy"].ge(params.occupancy_fraction)
            & result["down_repair"]
        ).fillna(False)
        result["short_entry"] = (
            close.lt(result["ma200"])
            & result["up_depth"].ge(params.depth_atr)
            & result["up_occupancy"].ge(params.occupancy_fraction)
            & result["up_repair"]
        ).fillna(False)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: BasinParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_entry"].to_numpy(dtype=bool)
        shorts = indicators["short_entry"].to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_atr = np.nan
        held = 0
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            if position != 0:
                # Yesterday's decision fills at today's open; ATR stays fixed.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                held += 1
                valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(close[i])
                stopped = valid and position * (close[i] - entry_price) <= -params.stop_atr * entry_atr
                if not valid or stopped or held >= params.hold_bars:
                    position = 0
                    entry_price = np.nan
                    entry_atr = np.nan
                    held = 0
            elif i >= warmup - 1 and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(close[i]):
                if longs[i]:
                    position = 1
                elif shorts[i]:
                    position = -1
                if position != 0:
                    entry_price = np.nan
                    entry_atr = atr[i]
                    held = 0
            signals[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
