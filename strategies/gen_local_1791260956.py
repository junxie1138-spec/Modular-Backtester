from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    compression_bars: int = 6
    reference_bars: int = 40
    width_ratio: float = 0.65
    downside_ratio: float = 0.40
    upside_retention: float = 0.75
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791260956"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.compression_bars, params.reference_bars) + 3)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if not 2 <= params.compression_bars < params.reference_bars:
            raise ValueError("Require 2 <= compression_bars < reference_bars")
        if not all(np.isfinite(x) and 0 < x <= 1 for x in (
            params.width_ratio, params.downside_ratio, params.upside_retention
        )):
            raise ValueError("Compression ratios must be finite and in (0, 1]")
        if params.hold_bars not in (3, 4, 5):
            raise ValueError("hold_bars must be 3, 4, or 5")

        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        # Compare today's return with envelopes completed yesterday.
        out["prior_return"] = out["return"].shift(1)
        out["near_floor"] = out["prior_return"].rolling(params.compression_bars).min()
        out["near_ceiling"] = out["prior_return"].rolling(params.compression_bars).max()
        out["reference_floor"] = out["prior_return"].rolling(params.reference_bars).min()
        out["reference_ceiling"] = out["prior_return"].rolling(params.reference_bars).max()
        out["near_width"] = out["near_ceiling"] - out["near_floor"]
        out["reference_width"] = out["reference_ceiling"] - out["reference_floor"]
        out["eligible"] = (
            (out["reference_floor"] < 0)
            & (out["reference_ceiling"] > 0)
            & (out["near_floor"] < 0)
            & (out["near_floor"].abs() <= params.downside_ratio * out["reference_floor"].abs())
            & (out["near_ceiling"] >= params.upside_retention * out["reference_ceiling"])
            & (out["near_width"] <= params.width_ratio * out["reference_width"])
            & (out["return"] > 0)
            & (out["return"] <= out["near_ceiling"])
        ).fillna(False)
        out["confirmed"] = out["eligible"] & out["eligible"].shift(1, fill_value=False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        if params.hold_bars not in (3, 4, 5):
            raise ValueError("hold_bars must be 3, 4, or 5")
        entries = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, entry in enumerate(entries):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
                signal[i] = 1
            elif entry:
                signal[i] = 1
                exit_bar = i + params.hold_bars

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=100)
    returns = np.tile([-0.02, 0.01, -0.002, 0.008, 0.003, -0.001], 17)[:100]
    close = 100.0 * np.cumprod(1.0 + returns)
    data = pd.DataFrame({
        "open": close, "high": close, "low": close,
        "close": close, "volume": 1000.0
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    ind = strategy.indicators(data, params)
    assert not ind["confirmed"].iloc[:params.reference_bars + 2].any()
    assert strategy.indicators(data.iloc[:70], params).equals(ind.iloc[:70])
    for hold in (3, 4, 5):
        params.hold_bars = hold
        forced = pd.DataFrame({"confirmed": False}, index=index)
        forced.iloc[2, 0] = True
        result = strategy.generate_signals(data, forced, ctx, params).data
        expected = np.zeros(len(data), dtype=int)
        expected[3:3 + hold] = 1
        assert np.array_equal(result["signal"].to_numpy(), expected)
        assert (result["size"] > 0).all()
    flat = data.copy()
    flat["close"] = 100.0
    assert not strategy.indicators(flat, params)["confirmed"].any()


if __name__ == "__main__":
    _self_check()
