from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeSeparationParams:
    range_window: int = 12
    overlap_window: int = 3
    atr_window: int = 14
    entry_overlap: float = 0.35
    trail_atr: float = 1.5
    hold_days: int = 4

    def __post_init__(self):
        if min(self.range_window, self.overlap_window, self.atr_window, self.hold_days) < 1:
            raise ValueError("Lookbacks and holding days must be positive.")
        if not 0.0 < self.entry_overlap < 0.60 or not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError("Invalid overlap threshold or ATR multiplier.")


class GeneratedStrategy(BaseStrategy[RangeSeparationParams]):
    strategy_id = "gen_local-w2_1791309406"

    @classmethod
    def params_type(cls):
        return RangeSeparationParams

    def warmup_bars(self, params: RangeSeparationParams) -> int:
        return int(max(params.range_window + 1, params.atr_window + 1, params.overlap_window + 2))

    def indicators(self, data: pd.DataFrame, params: RangeSeparationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_high"] = data["high"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["previous_close"] = data["close"].shift(1)
        out["range"] = (data["high"] - data["low"]).clip(lower=0)
        out["previous_range"] = out["range"].shift(1)
        out["baseline_range"] = out["range"].rolling(params.range_window, min_periods=params.range_window).mean().shift(1)
        out["intersection"] = (
            np.minimum(data["high"], out["previous_high"])
            - np.maximum(data["low"], out["previous_low"])
        ).clip(lower=0)
        out["overlap"] = out["intersection"].div(
            np.minimum(out["range"], out["previous_range"]).replace(0, np.nan)
        ).clip(0, 1)
        out["prior_overlap"] = out["overlap"].shift(1).rolling(
            params.overlap_window, min_periods=params.overlap_window
        ).mean()
        out["true_range"] = pd.concat([
            out["range"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out.loc[out["previous_close"].isna(), "true_range"] = np.nan
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["close_location"] = (data["close"] - data["low"]).div(out["range"].replace(0, np.nan))
        out["range_ratio"] = out["range"].div(out["baseline_range"].replace(0, np.nan))
        out["midpoint_move"] = (
            data["high"] + data["low"] - out["previous_high"] - out["previous_low"]
        ) / 2.0
        out["direction_strength"] = out["midpoint_move"].div(out["atr"].replace(0, np.nan)).clip(0, 1)
        out["confidence"] = (0.5 * (1.0 - out["overlap"]) + 0.5 * out["direction_strength"]).clip(0, 1)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out[["overlap", "prior_overlap", "atr", "range_ratio", "close_location", "confidence"]].notna().all(axis=1)
        out["entry"] = (
            ready
            & (out["atr"] > 0)
            & (out["prior_overlap"] >= 0.60)
            & (out["overlap"] <= params.entry_overlap)
            & (out["range_ratio"] >= 1.10)
            & (out["close_location"] >= 0.70)
            & (data["high"] > out["previous_high"])
            & (data["low"] > out["previous_low"])
            & (data["close"] > out["previous_close"])
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeSeparationParams,
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 0.25}, index=data.index)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        position_size = 0.25
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        confidence = indicators["confidence"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        for i in range(len(data)):
            valid = np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0
            if active:
                age += 1
                if valid:
                    high_water = max(high_water, closes[i])
                    stop = max(stop, high_water - params.trail_atr * atrs[i])
                if not valid or closes[i] <= stop or age >= params.hold_days:
                    active = False
                else:
                    signals[i] = 1
                    sizes[i] = position_size
            elif valid and entries[i] and np.isfinite(confidence[i]):
                active = True
                age = 0
                # The next bar is the entry; only its close and later closes enter the high-water mark.
                high_water = -np.inf
                stop = -np.inf
                position_size = float(np.clip(0.25 + 0.75 * confidence[i], 0.25, 1.0))
                signals[i] = 1
                sizes[i] = position_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"close": [100., 100., 102., 103., 101., 100., 100., 100.]}, index=index)
    ind = pd.DataFrame({
        "atr": [1., 1., 1., 1., 10., 10., 10., 10.],
        "entry": [False, True, False, False, False, False, False, False],
        "confidence": 0.8,
    }, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, ind, ctx, RangeSeparationParams(hold_days=10)).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    assert np.allclose(result.loc[result["signal"] == 1, "size"], 0.85)
    data["close"] = np.arange(100., 108.)
    ind["atr"] = 1.
    result = strategy.generate_signals(data, ind, ctx, RangeSeparationParams()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
