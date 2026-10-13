from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapFailureParams:
    ma_window: int = 30
    gap_window: int = 5
    atr_window: int = 14
    z_entry: float = 1.2
    min_gap_atr: float = 0.15
    failure_count: int = 2
    trail_atr: float = 2.5
    max_hold: int = 10
    cooldown: int = 2


class GeneratedStrategy(BaseStrategy[GapFailureParams]):
    strategy_id = "gen_local-w2_1790975405"

    @classmethod
    def params_type(cls):
        return GapFailureParams

    def warmup_bars(self, params: GapFailureParams) -> int:
        return int(max(params.ma_window, params.atr_window + params.gap_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapFailureParams) -> pd.DataFrame:
        if min(params.ma_window, params.gap_window, params.atr_window, params.failure_count, params.max_hold) < 1:
            raise ValueError("Windows, failure_count and max_hold must be positive.")
        if params.cooldown < 0 or not all(np.isfinite(v) and v > 0 for v in (params.z_entry, params.min_gap_atr, params.trail_atr)):
            raise ValueError("Thresholds must be finite and positive; cooldown must be nonnegative.")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["ma"] = data["close"].rolling(params.ma_window).mean()
        out["std"] = data["close"].rolling(params.ma_window).std(ddof=0)
        out["z"] = (data["close"] - out["ma"]) / out["std"].replace(0.0, np.nan)
        out["tr"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["tr"].rolling(params.atr_window).mean()
        out["previous_atr"] = out["atr"].shift(1).replace(0.0, np.nan)
        out["gap_atr"] = (data["open"] - out["previous_close"]) / out["previous_atr"]
        out["failed_down"] = ((out["gap_atr"] <= -params.min_gap_atr) & (data["close"] >= out["previous_close"])).astype(float)
        out["failed_up"] = ((out["gap_atr"] >= params.min_gap_atr) & (data["close"] <= out["previous_close"])).astype(float)
        valid_gap = np.isfinite(out["gap_atr"])
        out["failed_down"] = out["failed_down"].where(valid_gap)
        out["failed_up"] = out["failed_up"].where(valid_gap)
        out["failure_balance"] = (out["failed_down"] - out["failed_up"]).rolling(params.gap_window).sum()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapFailureParams) -> SignalFrame:
        long_entry = (indicators["z"] <= -params.z_entry) & (indicators["failure_balance"] >= params.failure_count)
        short_entry = (indicators["z"] >= params.z_entry) & (indicators["failure_balance"] <= -params.failure_count)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        decisions = np.zeros(len(data), dtype=int)
        position, age, blocked_until = 0, 0, -1
        watermark, stop = -np.inf, -np.inf
        values = np.column_stack([data["close"], indicators["atr"], indicators["z"], indicators["failure_balance"]])
        for i, (close, atr, z, balance) in enumerate(values):
            if not np.isfinite([close, atr, z, balance]).all() or close <= 0 or atr <= 0:
                position, age = 0, 0
                watermark, stop = -np.inf, -np.inf
                blocked_until = i + params.cooldown
                continue
            if position:
                age += 1
                # Signed prices give long highest-close and mirrored short lowest-close trails.
                # First update occurs on the fill bar, excluding the pre-entry decision close.
                signed_close = position * close
                watermark = max(watermark, signed_close)
                stop = max(stop, watermark - params.trail_atr * atr)
                if signed_close <= stop or age >= params.max_hold:
                    position, age = 0, 0
                    watermark, stop = -np.inf, -np.inf
                    blocked_until = i + params.cooldown
                    continue
            elif i > blocked_until and entries[i]:
                position, age = int(entries[i]), 0
                watermark, stop = -np.inf, -np.inf
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapFailureParams()
    index = pd.date_range("2020-01-01", periods=6)
    for side in (1, -1):
        close = np.array([100.0, 102.0, 104.0, 99.0, 99.0, 99.0])
        if side == -1:
            close = 200.0 - close
        data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0}, index=index)
        ind = pd.DataFrame({"atr": 1.0, "z": -2.0 * side, "failure_balance": 2.0 * side}, index=index)
        result = strategy.generate_signals(data, ind, None, params).data
        assert result["signal"].tolist() == [0, side, side, side, 0, 0]
        ind["failure_balance"] = 0.0
        assert not strategy.generate_signals(data, ind, None, params).data["signal"].any()
    params.max_hold = 2
    ind["z"], ind["failure_balance"] = -2.0, 2.0
    data["close"] = 100.0
    assert strategy.generate_signals(data, ind, None, params).data["signal"].tolist() == [0, 1, 1, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
