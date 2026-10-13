from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    streak_needed: int = 2
    rejection_fraction: float = 0.65
    hold_bars: int = 2
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791341438"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        if not 2 <= params.streak_needed <= 4:
            raise ValueError("streak_needed must be between 2 and 4")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        if not 0.5 < params.rejection_fraction < 1.0:
            raise ValueError("rejection_fraction must be between 0.5 and 1")
        if not np.isfinite(params.size) or not 0.0 < params.size <= 1.0:
            raise ValueError("size must be finite and in (0, 1]")
        return int(params.streak_needed + 2)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["previous_high"] = data["high"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["range"] = data["high"] - data["low"]
        out["previous_width"] = (
            out["previous_high"] - out["previous_low"]
        ).where(out["previous_high"] > out["previous_low"])
        out["inside"] = (
            (data["high"] <= out["previous_high"])
            & (data["low"] >= out["previous_low"])
            & (out["range"] > 0.0)
            & (out["range"] < out["previous_width"])
        ).fillna(False)
        resets = (~out["inside"]).cumsum()
        out["inside_streak"] = (
            out["inside"].astype(int).groupby(resets).cumsum()
            .clip(upper=params.streak_needed)
        )
        out["previous_streak"] = out["inside_streak"].shift(1).fillna(0)
        out["close_location"] = (
            (data["close"] - out["previous_low"]) / out["previous_width"]
        ).replace([np.inf, -np.inf], np.nan)
        out["ready"] = (
            (out["previous_streak"] >= params.streak_needed)
            & out["close_location"].between(0.0, 1.0)
        ).fillna(False)
        out["long_rejection"] = (
            out["ready"]
            & (data["low"] < out["previous_low"])
            & (data["high"] <= out["previous_high"])
            & (out["close_location"] >= params.rejection_fraction)
        ).fillna(False)
        out["short_rejection"] = (
            out["ready"]
            & (data["high"] > out["previous_high"])
            & (data["low"] >= out["previous_low"])
            & (out["close_location"] <= 1.0 - params.rejection_fraction)
        ).fillna(False)
        out["entry_direction"] = np.select(
            [out["long_rejection"], out["short_rejection"]],
            [1, -1], default=0,
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if position:
                if i == exit_bar:
                    position = 0
                raw[i] = position
                # Exit bars remain flat; entry signals cannot renew a position.
                continue
            if entry:
                position = int(entry)
                exit_bar = i + params.hold_bars
            raw[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    data = pd.DataFrame(
        {
            "open": [100.0] * 7,
            "high": [110.0, 108.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            "low": [90.0, 92.0, 94.0, 95.0, 94.0, 93.0, 92.0],
            "close": [100.0, 100.0, 100.0, 95.0, 100.0, 100.0, 100.0],
            "volume": [1000.0] * 7,
        },
        index=pd.date_range("2024-01-01", periods=7),
    )
    strategy = GeneratedStrategy()
    for direction in (-1, 1):
        sample = data.copy()
        if direction == 1:
            sample["high"] = 200.0 - data["low"]
            sample["low"] = 200.0 - data["high"]
            sample["close"] = 200.0 - data["close"]
        for hold in (1, 2):
            params = GeneratedParams(hold_bars=hold)
            features = strategy.indicators(sample, params)
            assert features["entry_direction"].tolist() == [0, 0, 0, direction, 0, 0, 0]
            # Opposing signals during a holding period must not trigger exits.
            features.loc[sample.index[4:], "entry_direction"] = -direction
            result = strategy.generate_signals(sample, features, None, params).data
            expected = [0, 0, 0, 0, direction, 0, -direction] if hold == 1 else [0, 0, 0, 0, direction, direction, 0]
            assert result["signal"].tolist() == expected
            assert result["size"].gt(0).all()
            assert strategy.warmup_bars(params) <= 10
