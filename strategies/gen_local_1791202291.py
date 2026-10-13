from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    window: int = 6
    breakout_z: float = 0.6
    size_gain: float = 0.8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791202291"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(2 * params.window + 2)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.window < 2:
            raise ValueError("window must be at least 2")
        if not np.isfinite(params.breakout_z) or params.breakout_z <= 0:
            raise ValueError("breakout_z must be finite and positive")
        if not np.isfinite(params.size_gain) or params.size_gain <= 0:
            raise ValueError("size_gain must be finite and positive")

        out = pd.DataFrame(index=data.index)
        opening = data["open"].where(data["open"] > 0)
        closing = data["close"].where(data["close"] > 0)
        out["gap"] = np.log(opening / closing.shift(1))
        out["body"] = np.log(closing / opening)
        out[["gap", "body"]] = out[["gap", "body"]].replace([np.inf, -np.inf], np.nan)
        n = params.window
        gap_window = out["gap"].rolling(n, min_periods=n)
        body_window = out["body"].rolling(n, min_periods=n)
        out["gap_mean"] = gap_window.mean().shift(1)
        out["body_mean"] = body_window.mean().shift(1)
        out["gap_variance"] = gap_window.var(ddof=0).shift(1)
        out["gap_body_covariance"] = gap_window.cov(out["body"], ddof=0).shift(1)
        out["beta"] = (
            out["gap_body_covariance"] / out["gap_variance"].where(out["gap_variance"] > 1e-12)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["expected_body"] = out["body_mean"] + out["beta"] * (out["gap"] - out["gap_mean"])
        out["residual"] = out["body"] - out["expected_body"]
        out["residual_volatility"] = out["residual"].rolling(n, min_periods=n).std(ddof=0).shift(1)
        out["score"] = (
            out["residual"] / out["residual_volatility"].where(out["residual_volatility"] > 1e-8)
        ).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        scores = indicators["score"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.15, dtype=float)
        position = 0
        entry_size = 0.15
        for i, score in enumerate(scores):
            direction = 0
            if np.isfinite(score):
                if score > params.breakout_z:
                    direction = 1
                elif score < -params.breakout_z:
                    direction = -1
            if position != 0:
                if direction == -position:
                    # An opposing breakout exits to flat; no same-bar re-entry.
                    position = 0
                    entry_size = 0.15
            elif direction != 0:
                position = direction
                excess = abs(score) - params.breakout_z
                entry_size = float(0.15 + 0.8 * np.tanh(params.size_gain * excess))
            signals[i] = position
            sizes[i] = entry_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.15).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = Params()
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"score": [np.nan, 1.0, 0.0, -1.0, -2.0, 0.0, 1.0, 1.0]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert result["size"].between(0.15, 0.95).all()
    assert result["size"].iloc[5] > result["size"].iloc[2]
    flat_indicators = strategy.indicators(data, params)
    flat_result = strategy.generate_signals(data, flat_indicators, None, params).data
    assert flat_result["signal"].eq(0).all()
    assert strategy.warmup_bars(params) == 14
