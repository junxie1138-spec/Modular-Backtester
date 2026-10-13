from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MonthBoundaryParams:
    volume_lookback: int = 20
    atr_lookback: int = 14
    volume_multiple: float = 1.25
    move_atr_threshold: float = 0.35
    trailing_atr: float = 0.9
    holding_days: int = 2


class GeneratedStrategy(BaseStrategy[MonthBoundaryParams]):
    strategy_id = "gen_local-w2_1790995903"

    @classmethod
    def params_type(cls) -> type[MonthBoundaryParams]:
        return MonthBoundaryParams

    def warmup_bars(self, params: MonthBoundaryParams) -> int:
        return int(max(1, params.volume_lookback, params.atr_lookback) + 1)

    def indicators(self, data: pd.DataFrame, params: MonthBoundaryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        previous_close = close.shift(1)
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_window = max(1, int(params.atr_lookback))
        volume_window = max(1, int(params.volume_lookback))
        out["atr"] = true_range.rolling(atr_window, min_periods=atr_window).mean()
        out["prior_volume_mean"] = data["volume"].astype(float).shift(1).rolling(
            volume_window, min_periods=volume_window
        ).mean()
        out["volume_ratio"] = data["volume"].div(
            out["prior_volume_mean"].where(out["prior_volume_mean"] > 0)
        )
        out["move_atr"] = close.diff().div(out["atr"].where(out["atr"] > 0))
        day = data.index.day
        out["calendar_direction"] = np.where(day >= 25, 1, np.where(day <= 3, -1, 0))
        valid = (
            np.isfinite(close)
            & np.isfinite(out["atr"])
            & (out["atr"] > 0)
            & np.isfinite(out["volume_ratio"])
            & np.isfinite(out["move_atr"])
        )
        participation = out["volume_ratio"] >= max(0.0, params.volume_multiple)
        threshold = max(0.0, params.move_atr_threshold)
        upward_move = out["move_atr"] > threshold
        downward_move = out["move_atr"] < -threshold
        long_entry = valid & participation & upward_move & (out["calendar_direction"] == 1)
        short_entry = valid & participation & downward_move & (out["calendar_direction"] == -1)
        out["entry_direction"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: MonthBoundaryParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        holding_days = min(2, max(1, int(params.holding_days)))
        distance = max(0.0, float(params.trailing_atr))
        position = 0
        age = 0
        high_water = np.nan
        stop_level = -np.inf

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and np.isfinite(atr) and atr > 0
            if position != 0:
                # This bar is held following the previous close's decision.
                age += 1
                if not valid:
                    position = 0
                else:
                    high_water = close if not np.isfinite(high_water) else max(high_water, close)
                    # The highest-close stop applies literally to both directions.
                    # It protects longs and covers shorts after a downward excursion.
                    stop_level = max(stop_level, high_water - distance * atr)
                    if close < stop_level or age >= holding_days:
                        position = 0
                signals[i] = position
                if position == 0:
                    age = 0
                    high_water = np.nan
                    stop_level = -np.inf
                # No same-close reentry after a stop or time exit.
                continue

            if valid and entries[i] != 0:
                position = int(entries[i])
                age = 0
                # Entry fills next bar; its first held close starts the watermark.
                high_water = np.nan
                stop_level = -np.inf
                signals[i] = position

        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
