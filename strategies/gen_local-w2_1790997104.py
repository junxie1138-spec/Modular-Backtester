from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    ma_window: int = 20
    vol_window: int = 20
    atr_window: int = 14
    efficiency_window: int = 10
    entry_z: float = 0.75
    maximum_z: float = 2.5
    minimum_efficiency: float = 0.25
    target_vol: float = 0.12
    strength_z: float = 2.0
    trail_atr: float = 1.8
    max_hold: int = 4
    minimum_size: float = 0.05
    maximum_size: float = 0.95


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1790997104"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.ma_window + 3, params.vol_window + 1,
                       params.atr_window + 1, params.efficiency_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["price_std"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["price_std"].replace(0.0, np.nan)
        out["abs_z"] = out["z"].abs()
        out["abs_z_previous"] = out["abs_z"].shift(1)
        out["abs_z_two_back"] = out["abs_z"].shift(2)
        out["abs_z_three_back"] = out["abs_z"].shift(3)
        out["z_previous"] = out["z"].shift(1)
        out["z_two_back"] = out["z"].shift(2)
        out["z_three_back"] = out["z"].shift(3)
        out["returns"] = close.pct_change(fill_method=None)
        out["annual_vol"] = out["returns"].rolling(
            params.vol_window, min_periods=params.vol_window
        ).std(ddof=0) * np.sqrt(252.0)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["close_change"] = close.diff()
        out["path_length"] = out["close_change"].abs().rolling(
            params.efficiency_window, min_periods=params.efficiency_window
        ).sum()
        out["net_change"] = close.diff(params.efficiency_window)
        out["efficiency"] = out["net_change"].abs() / out["path_length"].replace(0.0, np.nan)
        out["strength"] = (out["abs_z"] / params.strength_z).clip(0.0, 1.0)
        out["desired_size"] = (
            params.target_vol / out["annual_vol"].replace(0.0, np.nan)
            * out["strength"]
        ).clip(params.minimum_size, params.maximum_size)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        ind = indicators
        ready = ind[["z", "z_previous", "z_two_back", "z_three_back",
                     "annual_vol", "atr", "efficiency", "desired_size"]].notna().all(axis=1)
        ready &= (ind["annual_vol"] > 0.0) & (ind["atr"] > 0.0)
        paused = (
            (ind["abs_z_two_back"] < ind["abs_z_three_back"])
            & (ind["abs_z_previous"] < ind["abs_z_two_back"])
            & (ind["abs_z"] > ind["abs_z_previous"])
        )
        stable_direction = (
            (ind["z"] * ind["z_previous"] > 0.0)
            & (ind["z"] * ind["z_two_back"] > 0.0)
            & (ind["z"] * ind["z_three_back"] > 0.0)
        )
        entry = (
            ready & paused & stable_direction
            & (ind["abs_z"] >= params.entry_z)
            & (ind["abs_z"] <= params.maximum_z)
            & (ind["efficiency"] >= params.minimum_efficiency)
            & (ind["net_change"] * ind["z"] > 0.0)
        ).fillna(False)
        candidates = np.where(entry, np.where(ind["z"] > 0.0, 1, -1), 0)
        closes = data["close"].to_numpy(dtype=float)
        atrs = ind["atr"].to_numpy(dtype=float)
        sizes = ind["desired_size"].to_numpy(dtype=float)
        valid = ready.to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position_sizes = np.full(len(data), params.minimum_size, dtype=float)
        position = 0
        entry_bar = -1
        highest_close = 0.0
        lowest_close = 0.0
        high_stop = -np.inf
        short_stop = np.inf
        held_size = params.minimum_size

        for i in range(len(data)):
            close = closes[i]
            if position:
                if not valid[i] or not np.isfinite(close):
                    position = 0
                else:
                    highest_close = max(highest_close, close)
                    lowest_close = min(lowest_close, close)
                    # Required highest-close trail applies to both directions.
                    high_stop = max(high_stop, highest_close - params.trail_atr * atrs[i])
                    # Shorts also have a mirrored adverse-price trailing stop.
                    short_stop = min(short_stop, lowest_close + params.trail_atr * atrs[i])
                    exit_now = (
                        close <= high_stop
                        or (position == -1 and close >= short_stop)
                        or i - entry_bar >= params.max_hold
                    )
                    if exit_now:
                        position = 0
                signals[i] = position
                position_sizes[i] = held_size
                # An exit always produces a flat bar before another entry.
                continue

            if valid[i] and candidates[i] and np.isfinite(close):
                position = int(candidates[i])
                entry_bar = i
                highest_close = close
                lowest_close = close
                high_stop = close - params.trail_atr * atrs[i]
                short_stop = close + params.trail_atr * atrs[i]
                held_size = float(sizes[i])
                signals[i] = position
                position_sizes[i] = held_size

        df = pd.DataFrame({"signal": signals, "size": position_sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.minimum_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
