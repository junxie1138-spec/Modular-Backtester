from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeDecayParams:
    channel_bars: int = 40
    range_threshold: float = 1.3
    extension_threshold: float = 0.75
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[RangeDecayParams]):
    strategy_id = "gen_local-w2_1791222143"

    @classmethod
    def params_type(cls):
        return RangeDecayParams

    def warmup_bars(self, params: RangeDecayParams) -> int:
        if params.channel_bars < 2 or params.hold_bars < 1:
            raise ValueError("channel_bars must be >= 2 and hold_bars >= 1")
        thresholds = (params.range_threshold, params.extension_threshold)
        if not all(np.isfinite(x) and x > 0 for x in thresholds):
            raise ValueError("Thresholds must be finite and positive")
        return int(params.channel_bars + 7)

    def indicators(self, data: pd.DataFrame, params: RangeDecayParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["daily_range"] = (data["high"] - data["low"]).where(
            data["high"] > data["low"]
        )
        # Exclude the latest five sessions from the reference auction.
        out["reference_high"] = data["high"].shift(5).rolling(
            params.channel_bars, min_periods=params.channel_bars
        ).max()
        out["reference_low"] = data["low"].shift(5).rolling(
            params.channel_bars, min_periods=params.channel_bars
        ).min()
        out["reference_range"] = out["daily_range"].shift(5).rolling(
            params.channel_bars, min_periods=params.channel_bars
        ).median().replace(0.0, np.nan)
        out["range_ratio"] = out["daily_range"] / out["reference_range"]
        out["lower_overhang"] = (
            out["reference_low"] - data["low"]
        ).clip(lower=0.0) / out["reference_range"]
        out["upper_overhang"] = (
            data["high"] - out["reference_high"]
        ).clip(lower=0.0) / out["reference_range"]
        out = out.replace([np.inf, -np.inf], np.nan)
        out["contracting"] = out["daily_range"] < out["daily_range"].shift(1)
        elevated = out["range_ratio"] >= params.range_threshold
        out["long_setup"] = (
            out["contracting"] & elevated
            & (out["lower_overhang"] >= params.extension_threshold)
            & (out["upper_overhang"] == 0.0)
            & (data["close"] < out["reference_low"])
        ).fillna(False)
        out["short_setup"] = (
            out["contracting"] & elevated
            & (out["upper_overhang"] >= params.extension_threshold)
            & (out["lower_overhang"] == 0.0)
            & (data["close"] > out["reference_high"])
        ).fillna(False)
        out["long_confirmed"] = out["long_setup"] & out["long_setup"].shift(
            1, fill_value=False
        )
        out["short_confirmed"] = out["short_setup"] & out["short_setup"].shift(
            1, fill_value=False
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeDecayParams,
    ) -> SignalFrame:
        warmup = self.warmup_bars(params)
        longs = indicators["long_confirmed"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_confirmed"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i in range(warmup, len(data)):
            if position:
                if i == exit_bar:
                    position = 0
            elif longs[i] or shorts[i]:
                position = 1 if longs[i] else -1
                exit_bar = i + params.hold_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        # Shifting entry and exit equally preserves exactly hold_bars sessions.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = RangeDecayParams(channel_bars=2, hold_bars=3)
    index = pd.date_range("2020-01-01", periods=23, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.0, "volume": 1000.0}, index=index
    )
    strategy = GeneratedStrategy()
    ind = strategy.indicators(data, params)
    assert not ind["long_confirmed"].any()
    assert not ind["short_confirmed"].any()
    # Confirmed setups are injected to isolate holding-period accounting.
    ind.loc[index[10], "long_confirmed"] = True
    ind.loc[index[11:13], "short_confirmed"] = True
    ind.loc[index[15], "short_confirmed"] = True
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    actual = strategy.generate_signals(data, ind, ctx, params).data
    expected = np.zeros(len(data), dtype=int)
    expected[11:14] = 1
    expected[16:19] = -1
    assert np.array_equal(actual["signal"].to_numpy(), expected)
    assert (actual["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
