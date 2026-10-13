from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReservoirParams:
    peak_window: int = 20
    atr_window: int = 14
    vol_window: int = 20
    load_window: int = 8
    release_days: int = 2
    min_burden: float = 1.0
    release_threshold: float = 0.30
    min_depth: float = 0.40
    target_vol: float = 0.12
    trail_atr: float = 1.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[ReservoirParams]):
    strategy_id = "gen_local_1791256289"

    @classmethod
    def params_type(cls) -> type[ReservoirParams]:
        return ReservoirParams

    def warmup_bars(self, params: ReservoirParams) -> int:
        return int(max(params.peak_window, params.atr_window + 1, params.vol_window + 1) + params.load_window + params.release_days)

    def indicators(self, data: pd.DataFrame, params: ReservoirParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["peak"] = data["close"].rolling(params.peak_window, min_periods=params.peak_window).max()
        out["depth"] = (out["peak"] - data["close"]) / out["atr"].where(out["atr"] > 0)
        out["burden"] = out["depth"].rolling(params.load_window, min_periods=params.load_window).mean()
        out["released_depth"] = out["depth"].shift(params.release_days) - out["depth"]
        out["release"] = out["released_depth"] / out["burden"].where(out["burden"] > 0)
        out["returns"] = data["close"].pct_change(fill_method=None)
        out["volatility"] = out["returns"].rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0) * np.sqrt(252.0)
        out["strength"] = (out["burden"] / (1.0 + out["burden"])) * (out["release"] / (2.0 * params.release_threshold)).clip(0.0, 1.0)
        out["entry_size"] = (params.target_vol / out["volatility"].where(out["volatility"] > 0) * out["strength"]).clip(0.01, 0.95)
        out["entry"] = (
            (out["burden"] >= params.min_burden)
            & (out["depth"] >= params.min_depth)
            & (out["release"] >= params.release_threshold)
            & (out["release"].shift(1) < params.release_threshold)
            & (data["close"] > out["previous_close"])
            & out["entry_size"].notna()
        ).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ReservoirParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.01, dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        entry_sizes = indicators["entry_size"].to_numpy(dtype=float)
        active = False
        entry_bar = -1
        high_water = 0.0
        stop = 0.0
        position_size = 0.01
        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and np.isfinite(atr) and close > 0 and atr > 0
            if active:
                if not valid:
                    active = False
                else:
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.trail_atr * atr)
                    if close <= stop or i - entry_bar >= params.hold_bars:
                        active = False
            elif valid and entries[i] == 1 and np.isfinite(entry_sizes[i]):
                active = True
                entry_bar = i
                high_water = close
                stop = close - params.trail_atr * atr
                position_size = float(entry_sizes[i])
            if active:
                signals[i] = 1
                sizes[i] = position_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.01).astype(float)
        assert df["signal"].isin([0, 1]).all() and (df["size"] > 0).all()
        return SignalFrame(data=df, signal_column="signal", size_column="size")
