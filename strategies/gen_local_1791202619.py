from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    box_bars: int = 4
    atr_bars: int = 20
    roc_bars: int = 3
    compression_atr: float = 1.8
    acceleration_trigger: float = 0.15
    trail_atr: float = 1.5
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791202619"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.box_bars + 1, params.atr_bars + 1, params.roc_bars + 3))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["box_high"] = data["high"].rolling(params.box_bars).max()
        out["box_low"] = data["low"].rolling(params.box_bars).min()
        out["box_width"] = out["box_high"] - out["box_low"]
        out["previous_width"] = out["box_width"].shift(1)
        out["previous_high"] = out["box_high"].shift(1)
        out["width_atr"] = out["box_width"] / out["atr"].where(out["atr"] > 0)
        out["location"] = (close - out["box_low"]) / out["box_width"].where(out["box_width"] > 0)
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["roc_acceleration"] = out["roc"].diff()
        out["acceleration"] = out["roc_acceleration"] / (out["atr"] / close).where(out["atr"] > 0)
        out["previous_acceleration"] = out["acceleration"].shift(1)
        out["compression_depth"] = (1.0 - out["width_atr"] / params.compression_atr).clip(0.0, 1.0)
        out["strength"] = (
            (out["acceleration"] / (4.0 * params.acceleration_trigger)).clip(0.0, 1.0)
            * out["location"].clip(0.0, 1.0)
            * (0.5 + 0.5 * out["compression_depth"])
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["entry"] = (
            (out["width_atr"] < params.compression_atr)
            & (out["box_width"] <= out["previous_width"])
            & (out["location"] >= 0.65)
            & (close < out["previous_high"])
            & (out["roc"] > 0.0)
            & (out["acceleration"] > params.acceleration_trigger)
            & (out["previous_acceleration"] <= params.acceleration_trigger)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        active = False
        held = 0
        peak = stop = -np.inf
        trade_size = 0.25
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        strengths = indicators["strength"].fillna(0.0).to_numpy(dtype=float)

        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if active:
                held += 1
                if valid:
                    # Start the high-water mark on the actual next-bar entry day.
                    peak = max(peak, close)
                    stop = max(stop, peak - params.trail_atr * atr)
                if not valid or close <= stop or held >= params.max_hold:
                    active = False
                # This branch prevents re-entry on the same exit decision.
            elif valid and entries[i] and np.isfinite(strengths[i]):
                active = True
                held = 0
                peak = stop = -np.inf
                trade_size = float(0.25 + 0.70 * np.clip(strengths[i], 0.0, 1.0))
            if active:
                signals[i] = 1
                sizes[i] = trade_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GeneratedParams(max_hold=5)
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"close": [100, 101, 104, 103, 102, 101, 100, 101]}, index=index)
    derived = pd.DataFrame({
        "atr": [1, 1, 1, 10, 10, 1, 1, 1],
        "entry": [True, False, False, False, False, False, False, False],
        "strength": [0.8, 0, 0, 0, 0, 0, 0, 0],
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, derived, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert np.isclose(result["size"].iloc[1], 0.81)
    data["close"] = 100.0
    derived["atr"] = 1.0
    result = strategy.generate_signals(data, derived, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 1, 0, 0]
