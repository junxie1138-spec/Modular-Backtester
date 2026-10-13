from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownCrowdingParams:
    depth_window: int = 40
    percentile_window: int = 252
    extreme_percentile: float = 0.88
    occupancy_window: int = 4
    required_extreme_bars: int = 3
    atr_window: int = 14
    stop_atr: float = 2.0
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[DrawdownCrowdingParams]):
    strategy_id = "gen_local_1790995061"

    @classmethod
    def params_type(cls) -> type[DrawdownCrowdingParams]:
        return DrawdownCrowdingParams

    def warmup_bars(self, params: DrawdownCrowdingParams) -> int:
        return int(max(
            params.depth_window + params.percentile_window + params.occupancy_window + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: DrawdownCrowdingParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous_close = close.shift(1)
        out["previous_close"] = previous_close
        out["rolling_peak"] = close.rolling(params.depth_window, min_periods=params.depth_window).max()
        out["rolling_trough"] = close.rolling(params.depth_window, min_periods=params.depth_window).min()
        out["drawdown"] = 1.0 - close / out["rolling_peak"].replace(0.0, np.nan)
        out["runup"] = close / out["rolling_trough"].replace(0.0, np.nan) - 1.0
        out["drawdown_threshold"] = out["drawdown"].rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.extreme_percentile).shift(1)
        out["runup_threshold"] = out["runup"].rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.extreme_percentile).shift(1)
        out["deep"] = (
            out["drawdown"].gt(out["drawdown_threshold"])
            & out["drawdown_threshold"].gt(0.0)
        )
        out["elevated"] = (
            out["runup"].gt(out["runup_threshold"])
            & out["runup_threshold"].gt(0.0)
        )
        out["deep_occupancy"] = out["deep"].astype(float).rolling(
            params.occupancy_window, min_periods=params.occupancy_window
        ).sum()
        out["elevated_occupancy"] = out["elevated"].astype(float).rolling(
            params.occupancy_window, min_periods=params.occupancy_window
        ).sum()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["first_up_close"] = close.gt(previous_close) & previous_close.le(close.shift(2))
        out["first_down_close"] = close.lt(previous_close) & previous_close.ge(close.shift(2))
        out["ready"] = out[[
            "drawdown_threshold", "runup_threshold", "deep_occupancy",
            "elevated_occupancy", "atr", "previous_close",
        ]].notna().all(axis=1) & out["atr"].gt(0.0)
        out["long_entry"] = (
            out["ready"] & out["deep"] & out["first_up_close"]
            & out["deep_occupancy"].ge(params.required_extreme_bars)
        )
        out["short_entry"] = (
            out["ready"] & out["elevated"] & out["first_down_close"]
            & out["elevated_occupancy"].ge(params.required_extreme_bars)
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownCrowdingParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        long_entry = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        short_entry = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=np.int64)
        position = 0
        pending_entry = False
        entry_price = np.nan
        entry_atr = np.nan
        held_bars = 0

        for i in range(len(data)):
            if position != 0:
                # The preceding close's decision fills at this bar's open.
                if pending_entry:
                    entry_price = opens[i]
                    pending_entry = False
                held_bars += 1
                valid = (
                    np.isfinite(entry_price) and entry_price > 0.0
                    and np.isfinite(entry_atr) and entry_atr > 0.0
                    and np.isfinite(close[i])
                )
                stopped = False
                if valid:
                    # Entry price and entry ATR stay fixed throughout the trade.
                    if position == 1:
                        stopped = close[i] < entry_price - params.stop_atr * entry_atr
                    else:
                        stopped = close[i] > entry_price + params.stop_atr * entry_atr
                if not valid or stopped or held_bars >= params.holding_bars:
                    position = 0
                    held_bars = 0
                    entry_price = np.nan
                    entry_atr = np.nan
            elif np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0.0:
                if long_entry[i] and not short_entry[i]:
                    position = 1
                elif short_entry[i] and not long_entry[i]:
                    position = -1
                if position != 0:
                    pending_entry = True
                    entry_atr = atr[i]
                    held_bars = 0
            decisions[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
