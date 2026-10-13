from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAnalogParams:
    library_bars: int = 252
    vote_threshold: float = 0.25


class GeneratedStrategy(BaseStrategy[GapAnalogParams]):
    strategy_id = "gen_local_1791249861"

    @classmethod
    def params_type(cls):
        return GapAnalogParams

    @staticmethod
    def warmup_bars(params):
        return int(params.library_bars + 38)

    @staticmethod
    def indicators(data, params):
        if params.library_bars < 96 or not 0 < params.vote_threshold < 1:
            raise ValueError("library_bars must be >= 96; vote_threshold must be in (0, 1)")
        out = pd.DataFrame(index=data.index)
        prices = data[["open", "close"]].where(data[["open", "close"]] > 0)
        out["gap"] = np.log(prices["open"] / prices["close"].shift(1))
        out["gap_scale"] = out["gap"].rolling(20, min_periods=20).std().shift(1)
        out["gap_z"] = out["gap"] / out["gap_scale"].where(out["gap_scale"] > 1e-12)
        columns = []
        for lag in range(5):
            column = f"gap_lag_{lag}"
            out[column] = out["gap_z"].shift(lag)
            columns.append(column)
        out["mature_return"] = np.log(prices["open"] / prices["open"].shift(8))
        shapes = out[columns].to_numpy()
        returns = out["mature_return"].to_numpy()
        votes = np.full(len(data), np.nan)
        # ponytail: bounded brute-force analog search; index features if libraries become much larger.
        for t in range(34, len(data)):
            if not np.isfinite(shapes[t]).all():
                continue
            # An analog at j predicts open[j+1] to open[j+9]; both are known by t.
            candidates = np.arange(t - 9, max(24, t - 9 - params.library_bars), -8)
            valid = np.isfinite(shapes[candidates]).all(axis=1)
            valid &= np.isfinite(returns[candidates + 9])
            candidates = candidates[valid]
            if len(candidates) < 7:
                continue
            distances = np.square(shapes[candidates] - shapes[t]).sum(axis=1)
            nearest = candidates[np.argsort(distances, kind="stable")[:7]]
            votes[t] = np.sign(returns[nearest + 9]).mean()
        out["analog_vote"] = votes
        out["forecast"] = out["analog_vote"].rolling(5, min_periods=5).mean()
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        forecast = indicators["forecast"]
        conditions = np.select(
            [forecast >= params.vote_threshold, forecast <= -params.vote_threshold],
            [1, -1], default=0,
        )
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for t, condition in enumerate(conditions):
            if position == 0:
                position = int(condition)
            elif condition == -position:
                position = 0
            signals[t] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = GapAnalogParams(library_bars=96)
    index = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index))))
    opening = np.r_[100.0, close[:-1]] * np.exp(rng.normal(0, 0.004, len(index)))
    data = pd.DataFrame(
        {"open": opening, "high": np.maximum(opening, close) * 1.01,
         "low": np.minimum(opening, close) * 0.99, "close": close, "volume": 1e6},
        index=index,
    )
    full = GeneratedStrategy.indicators(data, params)
    prefix = GeneratedStrategy.indicators(data.iloc[:220], params)
    pd.testing.assert_frame_equal(full.iloc[:220], prefix)
    example = pd.DataFrame({"forecast": [1, 0, -1, -1, 0, 1, 1, 0]}, index=index[:8])
    result = GeneratedStrategy.generate_signals(data.iloc[:8], example, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    assert result["size"].gt(0).all()
