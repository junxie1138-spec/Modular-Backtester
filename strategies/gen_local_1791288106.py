from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    recent_bars: int = 4
    baseline_bars: int = 12
    dominance_threshold: float = 0.70


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791288106"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    @staticmethod
    def warmup_bars(params: StrategyParams) -> int:
        return int(params.recent_bars + params.baseline_bars + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.recent_bars < 1 or params.baseline_bars < 2:
            raise ValueError("Return windows must be positive, with baseline at least two bars")
        if not 0.5 < params.dominance_threshold <= 1.0:
            raise ValueError("Dominance threshold must be in (0.5, 1.0]")
        out = pd.DataFrame(index=data.index)
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return"] = returns.fillna(0.0)
        recent = pd.concat([returns.shift(j) for j in range(params.recent_bars)], axis=1).to_numpy(dtype=float)
        baseline = pd.concat([
            returns.shift(params.recent_bars + j)
            for j in range(params.baseline_bars)
        ], axis=1).to_numpy(dtype=float)
        valid = np.isfinite(recent).all(axis=1) & np.isfinite(baseline).all(axis=1)
        a = recent[:, :, None]
        b = baseline[:, None, :]
        dominance = ((a > b).astype(float) + 0.5 * (a == b)).mean(axis=(1, 2))
        recent_mean = np.where(np.isfinite(recent), recent, 0.0).mean(axis=1)
        out["dominance"] = np.where(valid, dominance, 0.5)
        out["recent_mean"] = np.where(valid, recent_mean, 0.0)
        upper = params.dominance_threshold
        lower = 1.0 - upper
        out["direction"] = np.select([
            valid & (dominance >= upper) & (recent_mean > 0.0),
            valid & (dominance <= lower) & (recent_mean < 0.0),
        ], [1, -1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        directions = indicators["direction"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(directions):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                # Opposing entry evidence closes the trade; reentry needs another bar.
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"close": np.full(8, 100.0)}, index=index)
    directions = pd.DataFrame({"direction": [1, 0, -1, -1, 0, 1, 1, 0]}, index=index)
    result = GeneratedStrategy.generate_signals(data, directions, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    mirrored = GeneratedStrategy.generate_signals(data, -directions, None, params).data
    assert mirrored["signal"].equals(-result["signal"])
    assert result["size"].gt(0).all()
    neutral = GeneratedStrategy.indicators(data, params)
    assert neutral["direction"].eq(0).all()
    assert not neutral.isna().any().any()
