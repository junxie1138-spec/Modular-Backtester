from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapClearanceParams:
    channel_bars: int = 8
    atr_bars: int = 14
    min_gap_atr: float = 0.35
    trail_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GapClearanceParams]):
    strategy_id = "gen_local_1791335235"

    @classmethod
    def params_type(cls):
        return GapClearanceParams

    def warmup_bars(self, params: GapClearanceParams) -> int:
        return int(max(params.channel_bars, params.atr_bars) + 1)

    def indicators(self, data: pd.DataFrame, params: GapClearanceParams) -> pd.DataFrame:
        if params.channel_bars < 1 or params.atr_bars < 1 or not 3 <= params.hold_bars <= 5:
            raise ValueError("Lookbacks must be positive and hold_bars must be 3 to 5.")
        if not np.isfinite([params.min_gap_atr, params.trail_atr]).all() or min(params.min_gap_atr, params.trail_atr) <= 0:
            raise ValueError("Gap and trailing ATR thresholds must be finite and positive.")
        previous = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["atr"] = true_range.rolling(params.atr_bars, min_periods=params.atr_bars).mean().shift(1)
        out["ceiling"] = data["high"].rolling(params.channel_bars, min_periods=params.channel_bars).max().shift(1)
        out["floor"] = data["low"].rolling(params.channel_bars, min_periods=params.channel_bars).min().shift(1)
        out["gap"] = data["open"] - previous
        out["gap_atr"] = out["gap"] / out["atr"].where(out["atr"] > 0)
        long_entry = (
            (out["gap_atr"] >= params.min_gap_atr)
            & (data["low"] > out["ceiling"])
            & (data["close"] > data["open"])
        )
        short_entry = (
            (out["gap_atr"] <= -params.min_gap_atr)
            & (data["high"] < out["floor"])
            & (data["close"] < data["open"])
        )
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0)
        out["clearance"] = np.where(out["gap"] > 0, data["open"] - out["ceiling"], out["floor"] - data["open"])
        out["clearance_share"] = (out["clearance"] / out["gap"].abs().where(out["gap"].abs() > 0)).clip(0.0, 1.0)
        out["entry_size"] = (0.25 + 0.70 * (out["gap_atr"].abs() / 2.0).clip(0.0, 1.0) * out["clearance_share"]).fillna(0.25)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapClearanceParams) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.25, dtype=float)
        side, age = 0, 0
        high_water, stop, exposure = 0.0, 0.0, 0.25
        values = indicators[["atr", "entry", "entry_size"]].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        for i in range(n):
            close = closes[i]
            atr, entry, entry_size = values[i]
            if not np.isfinite([close, atr, entry, entry_size]).all() or close <= 0 or atr <= 0 or entry_size <= 0:
                side = 0
                continue
            if side:
                age += 1
                # Signed closes give shorts the symmetric low-water trailing stop.
                high_water = max(high_water, side * close)
                stop = max(stop, high_water - params.trail_atr * atr)
                if side * close <= stop or age >= params.hold_bars:
                    side = 0
            elif entry:
                side, age = int(entry), 0
                high_water = side * close
                stop = high_water - params.trail_atr * atr
                exposure = float(np.clip(entry_size, 0.25, 0.95))
            signals[i] = side
            sizes[i] = exposure if side else 0.25
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapClearanceParams()
    index = pd.date_range("2020-01-01", periods=6)
    for side in (1, -1):
        closes = 100.0 + side * np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0])
        data = pd.DataFrame({"close": closes}, index=index)
        derived = pd.DataFrame({
            "atr": [1.0, 1.0, 4.0, 1.0, 1.0, 1.0],
            "entry": [side, 0, 0, 0, 0, 0],
            "entry_size": [0.8] * 6,
        }, index=index)
        result = strategy.generate_signals(data, derived, None, params).data
        assert result["signal"].tolist() == [0, side, side, 0, 0, 0]
        assert result["size"].iloc[1] == 0.8
        data["close"] = 100.0
        derived["atr"] = 1.0
        result = strategy.generate_signals(data, derived, None, params).data
        assert result["signal"].tolist() == [0, side, side, side, side, 0]
        assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
