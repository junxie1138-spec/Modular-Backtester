from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRejectionParams:
    z_entry: float = 1.5
    trail_atr: float = 1.0


class GeneratedStrategy(BaseStrategy[GapRejectionParams]):
    strategy_id = "gen_local_1791326934"

    @classmethod
    def params_type(cls) -> type[GapRejectionParams]:
        return GapRejectionParams

    def warmup_bars(self, params: GapRejectionParams) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: GapRejectionParams) -> pd.DataFrame:
        if not np.isfinite(params.z_entry) or params.z_entry <= 0:
            raise ValueError("z_entry must be finite and positive")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        close = data["close"]
        previous = close.shift(1)
        mean = close.rolling(20, min_periods=20).mean().shift(1)
        scale = close.rolling(20, min_periods=20).std(ddof=0).shift(1)
        scale = scale.where(scale > 0)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        true_range = true_range.where(previous.notna())
        out = pd.DataFrame(index=data.index)
        out["mean"] = mean
        out["scale"] = scale
        out["previous_z"] = (previous - mean) / scale
        out["open_z"] = (data["open"] - mean) / scale
        out["close_z"] = (close - mean) / scale
        out["gap_z"] = (data["open"] - previous) / scale
        out["rejection_z"] = (close - data["open"]) / scale
        out["atr"] = true_range.rolling(14, min_periods=14).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        long_entry = (
            (out["previous_z"] <= -params.z_entry)
            & (out["gap_z"] <= -0.25)
            & (out["rejection_z"] >= -0.5 * out["gap_z"])
            & (out["close_z"] <= -params.z_entry)
        )
        short_entry = (
            (out["previous_z"] >= params.z_entry)
            & (out["gap_z"] >= 0.25)
            & (out["rejection_z"] <= -0.5 * out["gap_z"])
            & (out["close_z"] >= params.z_entry)
        )
        valid = out.notna().all(axis=1) & (out["atr"] > 0)
        out["entry"] = np.select(
            [valid & long_entry, valid & short_entry], [1, -1], default=0
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRejectionParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        highest = -np.inf
        lowest = np.inf
        lower_stop = -np.inf
        upper_stop = np.inf
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0
            if position:
                age += 1
                if valid:
                    # These closes occur after the next-open entry fill.
                    highest = max(highest, close[i])
                    lowest = min(lowest, close[i])
                    lower_stop = max(lower_stop, highest - params.trail_atr * atr[i])
                    upper_stop = min(upper_stop, lowest + params.trail_atr * atr[i])
                # The high-water stop applies to either direction; shorts also
                # have a mirrored low-water stop against adverse advances.
                stopped = valid and (
                    close[i] <= lower_stop
                    or (position == -1 and close[i] >= upper_stop)
                )
                if not valid or stopped or age >= 2:
                    position = 0
                signal[i] = position
                continue
            if valid and entries[i]:
                position = int(entries[i])
                age = 0
                highest = -np.inf
                lowest = np.inf
                lower_stop = -np.inf
                upper_stop = np.inf
            signal[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GapRejectionParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100.0, 100.0, 98.0, 100.0, 100.0, 100.0]}, index=index)
    derived = pd.DataFrame({"atr": [1.0] * 6, "entry": [1, 0, 0, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    assert (result["size"] > 0).all()
    derived["entry"] = [-1, 0, 0, 0, 0, 0]
    data["close"] = [100.0, 100.0, 102.0, 100.0, 100.0, 100.0]
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, -1, -1, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
