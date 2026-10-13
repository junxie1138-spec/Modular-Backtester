from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReturnFloorParams:
    corridor_bars: int = 30
    clearance_bars: int = 4
    clearance_threshold: float = 0.60
    atr_bars: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[ReturnFloorParams]):
    strategy_id = "gen_local_1790893676"

    @classmethod
    def params_type(cls) -> type[ReturnFloorParams]:
        return ReturnFloorParams

    @staticmethod
    def warmup_bars(params: ReturnFloorParams) -> int:
        return int(max(params.corridor_bars + params.clearance_bars + 1, params.atr_bars + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: ReturnFloorParams) -> pd.DataFrame:
        if min(params.corridor_bars, params.clearance_bars, params.atr_bars, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive")
        if not 0.0 < params.clearance_threshold < 1.0:
            raise ValueError("Clearance threshold must lie between zero and one")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0:
            raise ValueError("Stop multiplier must be finite and positive")
        if not 0.0 < params.min_size <= params.max_size <= 1.0:
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0.0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        prior = out["return"].shift(params.clearance_bars)
        out["corridor_low"] = prior.rolling(params.corridor_bars).min()
        out["corridor_high"] = prior.rolling(params.corridor_bars).max()
        out["corridor_width"] = (out["corridor_high"] - out["corridor_low"]).where(lambda x: x > 0.0)
        out["recent_floor"] = out["return"].rolling(params.clearance_bars).min()
        out["recent_ceiling"] = out["return"].rolling(params.clearance_bars).max()
        out["recent_mean"] = out["return"].rolling(params.clearance_bars).mean()
        out["floor_position"] = (out["recent_floor"] - out["corridor_low"]) / out["corridor_width"]
        out["ceiling_position"] = (out["recent_ceiling"] - out["corridor_low"]) / out["corridor_width"]
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        required = out[["floor_position", "ceiling_position", "recent_mean", "atr"]]
        out["ready"] = np.isfinite(required).all(axis=1) & out["atr"].gt(0.0)
        out["strength"] = out["floor_position"].clip(0.0, 1.0).where(out["ready"], 0.0).fillna(0.0)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ReturnFloorParams) -> SignalFrame:
        floor = indicators["floor_position"]
        entries = (
            indicators["ready"]
            & floor.ge(params.clearance_threshold)
            & floor.shift(1).lt(params.clearance_threshold)
            & indicators["ceiling_position"].le(1.0)
            & indicators["recent_mean"].gt(0.0)
        ).fillna(False).to_numpy(dtype=bool)
        strengths = indicators["strength"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        active = False
        pending_fill = False
        fill_bar = -1
        entry_atr = 0.0
        stop_price = 0.0
        held_size = params.min_size
        for i in range(len(data)):
            exited = False
            if active and pending_fill:
                pending_fill = False
                fill_bar = i
                if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                    active = False
                    exited = True
                else:
                    # Freeze decision-bar ATR against the next bar's entry open.
                    stop_price = opens[i] - params.stop_atr * entry_atr
            if active:
                stop_hit = np.isfinite(closes[i]) and closes[i] < stop_price
                expired = i - fill_bar + 1 >= params.hold_bars
                if stop_hit or expired:
                    active = False
                    exited = True
            if not active and not exited and entries[i]:
                active = True
                pending_fill = True
                entry_atr = atr[i]
                held_size = params.min_size + (params.max_size - params.min_size) * strengths[i]
            if active:
                signals[i] = 1
                sizes[i] = held_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = ReturnFloorParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100.0, 100.0, 101.0, 100.0, 99.0, 100.0],
        "high": [102.0] * 6,
        "low": [97.0] * 6,
        "close": [100.0, 100.0, 100.0, 98.0, 100.0, 100.0],
        "volume": [1000.0] * 6,
    }, index=index)
    ind = pd.DataFrame({
        "floor_position": [0.2, 0.7, 0.7, 0.7, 0.7, 0.7],
        "ceiling_position": [0.9] * 6,
        "recent_mean": [0.01] * 6,
        "atr": [1.0, 1.0, 10.0, 10.0, 10.0, 10.0],
        "ready": [True] * 6,
        "strength": [0.2, 0.7, 0.7, 0.7, 0.7, 0.7],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, ind, None, p).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    assert np.isclose(result["size"].iloc[2], 0.74)
    assert result["size"].gt(0.0).all()
    expired = GeneratedStrategy.generate_signals(data, ind, None, ReturnFloorParams(hold_bars=1)).data
    assert expired["signal"].tolist() == [0, 0, 1, 0, 0, 0]
    derived = GeneratedStrategy.indicators(data, p)
    assert not derived["ready"].any()
    assert GeneratedStrategy.warmup_bars(p) >= p.corridor_bars + p.clearance_bars + 1
