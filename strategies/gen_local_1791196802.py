from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    ma_window: int = 40
    edge_window: int = 5
    atr_window: int = 14
    z_entry: float = 1.0
    edge_threshold: float = 0.2
    trail_atr: float = 3.0
    max_hold: int = 20


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791196802"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.ma_window, 2 * params.edge_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.ma_window, params.edge_window, params.atr_window, params.max_hold) < 1:
            raise ValueError("Lookbacks and holding period must be positive")
        if params.ma_window < 2 or min(params.z_entry, params.trail_atr) <= 0 or params.edge_threshold < 0:
            raise ValueError("Invalid signal or trailing-stop parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_window).mean()
        out["price_std"] = close.rolling(params.ma_window).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["price_std"].replace(0.0, np.nan)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["range_high"] = data["high"].rolling(params.edge_window).max()
        out["range_low"] = data["low"].rolling(params.edge_window).min()
        denominator = out["atr"].replace(0.0, np.nan)
        out["high_migration"] = (out["range_high"] - out["range_high"].shift(params.edge_window)) / denominator
        out["low_migration"] = (out["range_low"] - out["range_low"].shift(params.edge_window)) / denominator
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        valid = indicators[["z", "atr", "high_migration", "low_migration"]].notna().all(axis=1)
        valid &= indicators["atr"].gt(0) & data["close"].gt(0)
        valid &= np.isfinite(data["close"])
        long_entry = valid & indicators["z"].ge(params.z_entry)
        long_entry &= indicators["high_migration"].gt(params.edge_threshold) & indicators["low_migration"].gt(params.edge_threshold)
        short_entry = valid & indicators["z"].le(-params.z_entry)
        short_entry &= indicators["high_migration"].lt(-params.edge_threshold) & indicators["low_migration"].lt(-params.edge_threshold)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        usable = valid.to_numpy(dtype=bool)
        longs = long_entry.to_numpy(dtype=bool)
        shorts = short_entry.to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        extreme = 0.0
        stop = 0.0
        for i in range(len(data)):
            if not usable[i]:
                position = 0
                continue
            close = closes[i]
            atr = atrs[i]
            if position:
                expired = i - entry_bar >= params.max_hold
                if position == 1:
                    extreme = max(extreme, close)
                    stop = max(stop, extreme - params.trail_atr * atr)
                    stopped = close <= stop
                else:
                    extreme = min(extreme, close)
                    stop = min(stop, extreme + params.trail_atr * atr)
                    stopped = close >= stop
                if expired or stopped:
                    position = 0
                signals[i] = position
                continue
            if longs[i] or shorts[i]:
                position = 1 if longs[i] else -1
                entry_bar = i
                extreme = close
                stop = close - position * params.trail_atr * atr
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        assert df["signal"].isin([-1, 0, 1]).all() and df["size"].gt(0).all()
        return SignalFrame(data=df, signal_column="signal", size_column="size")
