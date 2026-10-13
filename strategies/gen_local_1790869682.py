from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 20
    complexity_window: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790869682"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.lookback, 2 * params.complexity_window + 2, 4))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.lookback < 4 or params.complexity_window < 3:
            raise ValueError("lookback must be >= 4 and complexity_window >= 3")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        out["close"] = close
        out["lag2"] = close.shift(2)
        out["lag1"] = close.shift(1)
        valid = out[["close", "lag1", "lag2"]].notna().all(axis=1)
        # Equal prices retain chronological order, giving six stable ordinal patterns.
        out["pattern"] = (
            4 * (out["lag2"] > out["lag1"]).astype(int)
            + 2 * (out["lag2"] > close).astype(int)
            + (out["lag1"] > close).astype(int)
        ).where(valid)
        out["entropy"] = 0.0
        for pattern in (0, 1, 3, 4, 6, 7):
            name = f"p{pattern}"
            out[name] = out["pattern"].eq(pattern).astype(float).where(valid).rolling(
                params.complexity_window, min_periods=params.complexity_window
            ).mean()
            out["entropy"] -= out[name] * np.log(out[name].clip(lower=1e-12))
        out["entropy"] /= np.log(6.0)
        out["prior_entropy"] = out["entropy"].shift(params.complexity_window)
        out["entropy_gain"] = out["entropy"] - out["prior_entropy"]
        out["peak"] = close.rolling(params.lookback, min_periods=params.lookback).max()
        out["trough"] = close.rolling(params.lookback, min_periods=params.lookback).min()
        out["span"] = (out["peak"] - out["trough"]).replace(0.0, np.nan)
        out["position"] = (close - out["trough"]) / out["span"]
        out["range_fraction"] = out["span"] / close
        out["move"] = close.diff(3)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        active = (indicators["entropy_gain"] > 0.08) & (indicators["range_fraction"] > 0.015)
        long_entry = active & indicators["position"].between(0.20, 0.50) & (indicators["move"] > 0)
        short_entry = active & indicators["position"].between(0.50, 0.80) & (indicators["move"] < 0)
        desire = np.select([long_entry, short_entry], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(desire):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0  # Opposite qualification exits; re-entry requires a later bar.
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=80)
    close = 100 + 4 * np.sin(np.arange(80) / 3.0)
    data = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": 1000.0}, index=index
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    full = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(full.iloc[:50], strategy.indicators(data.iloc[:50], params))
    assert full["prior_entropy"].first_valid_index() == index[2 * params.complexity_window + 1]
    probe = pd.DataFrame(
        {"entropy_gain": 0.2, "range_fraction": 0.03,
         "position": [0.3, 0.9, 0.7, 0.7, 0.9, 0.3, 0.9],
         "move": [1, 0, -1, -1, 0, 1, 0]}, index=index[:7]
    )
    result = strategy.generate_signals(data.iloc[:7], probe, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
