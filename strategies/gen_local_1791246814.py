from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    box_bars: int = 4
    range_bars: int = 20
    compression_limit: float = 1.7
    min_gap: float = 0.35
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791246814"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.box_bars, params.range_bars) + 2

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.box_bars < 1 or params.range_bars < 1:
            raise ValueError("Lookbacks must be positive integers")
        if not np.isfinite(params.compression_limit) or params.compression_limit <= 0:
            raise ValueError("compression_limit must be finite and positive")
        if not np.isfinite(params.min_gap) or params.min_gap <= 0:
            raise ValueError("min_gap must be finite and positive")

        out = pd.DataFrame(index=data.index)
        out["daily_range"] = data["high"] - data["low"]
        out["reference_range"] = out["daily_range"].rolling(params.range_bars).mean().shift(1).where(lambda s: s > 0)
        out["box_high"] = data["high"].rolling(params.box_bars).max().shift(1)
        out["box_low"] = data["low"].rolling(params.box_bars).min().shift(1)
        out["anchor"] = data["close"].shift(1)
        out["gap_units"] = (data["open"] - out["anchor"]) / out["reference_range"]
        out["compression"] = (out["box_high"] - out["box_low"]) / out["reference_range"]
        out["compressed"] = out["compression"].le(params.compression_limit)
        out["inside_box"] = data["close"].between(out["box_low"], out["box_high"])

        long_reentry = (
            out["compressed"] & out["inside_box"]
            & out["gap_units"].le(-params.min_gap)
            & data["open"].lt(out["box_low"])
            & data["close"].gt(data["open"])
        )
        short_reentry = (
            out["compressed"] & out["inside_box"]
            & out["gap_units"].ge(params.min_gap)
            & data["open"].gt(out["box_high"])
            & data["close"].lt(data["open"])
        )
        out["reentry_side"] = np.select([long_reentry, short_reentry], [1, -1], default=0)

        # Second close must restore the pre-gap anchor inside the same frozen box.
        out["inside_frozen_box"] = data["close"].between(out["box_low"].shift(1), out["box_high"].shift(1))
        long_confirm = (
            out["reentry_side"].shift(1).eq(1) & out["inside_frozen_box"]
            & data["close"].gt(out["anchor"].shift(1))
            & data["close"].gt(data["close"].shift(1))
        )
        short_confirm = (
            out["reentry_side"].shift(1).eq(-1) & out["inside_frozen_box"]
            & data["close"].lt(out["anchor"].shift(1))
            & data["close"].lt(data["close"].shift(1))
        )
        out["confirmed_side"] = np.select([long_confirm, short_confirm], [1, -1], default=0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")

        candidates = indicators["confirmed_side"].fillna(0).to_numpy(dtype=int)
        raw_signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_at = -1
        for i, candidate in enumerate(candidates):
            if position:
                # No candidate can shorten or extend the scheduled holding period.
                if i == exit_at:
                    position = 0
            elif candidate:
                position = int(candidate)
                exit_at = i + params.hold_bars
            raw_signal[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw_signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000000.0},
        index=pd.date_range("2020-01-01", periods=29, freq="B"),
    )
    data.iloc[24] = [98.5, 100.5, 98.0, 99.8, 1000000.0]
    data.iloc[25] = [99.9, 100.8, 99.7, 100.4, 1000000.0]
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    indicators = strategy.indicators(data, params)
    assert indicators["reentry_side"].iloc[24] == 1
    assert indicators["confirmed_side"].iloc[24:26].tolist() == [0, 1]
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[24:29].tolist() == [0, 0, 1, 1, 0]

    candidates = pd.DataFrame(
        {"confirmed_side": [0, 1, -1, -1, 0, -1, 1, 0, 0]},
        index=data.index[:9],
    )
    result = strategy.generate_signals(data.iloc[:9], candidates, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, -1, -1, 0]


if __name__ == "__main__":
    _self_check()
