from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalStrengthParams:
    fractional_order: float = 0.45
    entry_strength: float = 1.0
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[FractionalStrengthParams]):
    strategy_id = "gen_local_1790942153"

    @classmethod
    def params_type(cls) -> type[FractionalStrengthParams]:
        return FractionalStrengthParams

    @staticmethod
    def warmup_bars(params: FractionalStrengthParams) -> int:
        return 9

    @staticmethod
    def indicators(data: pd.DataFrame, params: FractionalStrengthParams) -> pd.DataFrame:
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one")
        if not np.isfinite(params.entry_strength) or params.entry_strength <= 0.0:
            raise ValueError("entry_strength must be finite and positive")
        close = data["close"].astype(float)
        weights = np.ones(6, dtype=float)
        for k in range(1, 6):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1) / k
        out = pd.DataFrame(index=data.index)
        out["fractional_close"] = sum(weights[k] * close.shift(k) for k in range(6))
        out["anchored_impulse"] = out["fractional_close"] - weights.sum() * close.shift(5)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["mean_range"] = out["true_range"].rolling(6, min_periods=6).mean()
        out["fractional_strength"] = out["anchored_impulse"] / out["mean_range"].where(out["mean_range"] > 0.0)
        out["impulse_amplitude"] = out["anchored_impulse"].abs()
        out["amplitude_baseline"] = out["impulse_amplitude"].rolling(3, min_periods=3).mean().shift(1)
        out["direction"] = np.sign(out["anchored_impulse"])
        out["direction_votes"] = out["direction"].rolling(3, min_periods=3).sum()
        eligible = (
            out["fractional_strength"].abs().ge(params.entry_strength)
            & out["impulse_amplitude"].gt(out["amplitude_baseline"])
            & out["direction_votes"].abs().eq(3.0)
        )
        out["entry_side"] = out["direction"].where(eligible, 0.0)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FractionalStrengthParams,
    ) -> SignalFrame:
        if params.holding_bars < 1:
            raise ValueError("holding_bars must be positive")
        candidates = indicators["entry_side"].fillna(0.0).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for i, candidate in enumerate(candidates):
            if position != 0:
                if i - entry_bar == params.holding_bars:
                    position = 0
            elif candidate != 0:
                position = 1 if candidate > 0 else -1
                entry_bar = i
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=10, freq="B")
    sample = pd.DataFrame(index=index)
    entries = pd.DataFrame({"entry_side": [1.0] * 7 + [-1.0] * 3}, index=index)
    result = GeneratedStrategy.generate_signals(
        sample, entries, None, FractionalStrengthParams(holding_bars=6)
    ).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 1, 1, 0, -1, -1]
    assert result["size"].gt(0.0).all()
