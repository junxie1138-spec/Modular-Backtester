from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    peak_period: int = 20
    atr_period: int = 14
    depth_atr: float = 2.0
    breakeven_pct: float = 0.01
    trail_atr: float = 1.5
    initial_stop_atr: float = 2.5
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = "gen_local_1790852121"

    @classmethod
    def params_type(cls) -> type[DrawdownParams]:
        return DrawdownParams

    @staticmethod
    def warmup_bars(params: DrawdownParams) -> int:
        return int(max(200, params.peak_period + 1, params.atr_period + 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        if params.peak_period < 2 or params.atr_period < 1:
            raise ValueError("Lookbacks must be positive; peak_period must be at least two.")
        if not all(np.isfinite(x) and x > 0 for x in (params.depth_atr, params.breakeven_pct, params.trail_atr, params.initial_stop_atr)):
            raise ValueError("Depth and stop parameters must be finite and positive.")
        if params.max_hold < 1:
            raise ValueError("max_hold must be positive.")
        close = data["close"]
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        true_range = true_range.where(previous.notna())
        atr = true_range.rolling(params.atr_period, min_periods=params.atr_period).mean()
        peak = data["high"].rolling(params.peak_period, min_periods=params.peak_period).max()
        depth = (peak - close) / atr.where(atr > 0)
        return pd.DataFrame({
            "ma200": close.rolling(200, min_periods=200).mean(),
            "atr": atr,
            "peak": peak,
            "depth": depth,
            "previous_depth": depth.shift(1),
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DrawdownParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        depth = indicators["depth"].to_numpy(dtype=float)
        previous_depth = indicators["previous_depth"].to_numpy(dtype=float)
        ready = np.isfinite(close) & np.isfinite(atr) & (atr > 0) & np.isfinite(ma) & np.isfinite(depth) & np.isfinite(previous_depth)
        long_entry = ready & (close > ma) & (previous_depth < params.depth_atr) & (depth >= params.depth_atr)
        short_entry = ready & (close < ma) & (previous_depth > params.depth_atr) & (depth <= params.depth_atr)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry = 0.0
        stop = 0.0
        armed = False
        held = 0

        for i in range(len(data)):
            # Yesterday's decision enters at today's open.
            if position == 0 and i > 0 and signals[i - 1] != 0:
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    continue
                position = int(signals[i - 1])
                entry = opens[i]
                stop = position * entry - params.initial_stop_atr * atr[i - 1]
                armed = False
                held = 0

            if position != 0:
                held += 1
                if not ready[i]:
                    position = 0
                    continue
                signed_close = position * close[i]
                signed_entry = position * entry
                if position * (close[i] / entry - 1.0) >= params.breakeven_pct:
                    armed = True
                if armed:
                    # Favorable-price coordinates make both directions ratchet upward.
                    stop = max(stop, signed_entry, signed_close - params.trail_atr * atr[i])
                regime_lost = position * (close[i] - ma[i]) <= 0
                if signed_close <= stop or regime_lost or held >= params.max_hold:
                    position = 0
                signals[i] = position
                continue

            if long_entry[i]:
                signals[i] = 1
            elif short_entry[i]:
                signals[i] = -1

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = DrawdownParams()
    index = pd.date_range("2020-01-01", periods=7)
    for direction in (1, -1):
        close = np.array([100, 98, 99, 101, 99, 99, 99], dtype=float)
        opens = np.array([100, 100, 98, 99, 101, 99, 99], dtype=float)
        depth = np.array([1, 2.1, 1.5, 1, 1, 1, 1], dtype=float)
        if direction == -1:
            close = 200 - close
            opens = 200 - opens
            depth = 4 - depth
        data = pd.DataFrame({"open": opens, "high": np.maximum(opens, close) + 1, "low": np.minimum(opens, close) - 1, "close": close, "volume": 1000.0}, index=index)
        derived = pd.DataFrame({"ma200": 90.0 if direction == 1 else 110.0, "atr": 1.0, "depth": depth, "previous_depth": pd.Series(depth, index=index).shift(1)}, index=index)
        result = GeneratedStrategy.generate_signals(data, derived, None, params).data
        assert result["signal"].tolist() == [0, 0, direction, direction, direction, 0, 0]
        assert (result["size"] > 0).all()
    assert GeneratedStrategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
