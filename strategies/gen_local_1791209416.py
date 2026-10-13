from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalSeasonParams:
    pattern_window: int = 15
    complexity_cap: float = 0.85
    min_bias: float = 0.20


class GeneratedStrategy(BaseStrategy[OrdinalSeasonParams]):
    strategy_id = "gen_local_1791209416"

    @classmethod
    def params_type(cls):
        return OrdinalSeasonParams

    def warmup_bars(self, params: OrdinalSeasonParams) -> int:
        return int(params.pattern_window) + 2

    def indicators(self, data: pd.DataFrame, params: OrdinalSeasonParams) -> pd.DataFrame:
        if params.pattern_window < 3:
            raise ValueError("pattern_window must be at least 3")
        if not 0.0 < params.complexity_cap <= 1.0:
            raise ValueError("complexity_cap must be in (0, 1]")
        if not 0.0 <= params.min_bias < 1.0:
            raise ValueError("min_bias must be in [0, 1)")
        out = pd.DataFrame(index=data.index)
        out["oldest"] = data["close"].shift(2)
        out["middle"] = data["close"].shift(1)
        out["newest"] = data["close"]
        a, b, c = out["oldest"], out["middle"], out["newest"]
        out["pattern"] = np.select(
            [
                (a < b) & (b < c),
                (a < c) & (c < b),
                (b < a) & (a < c),
                (b < c) & (c < a),
                (c < a) & (a < b),
                (c < b) & (b < a),
            ],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            default=np.nan,
        )
        valid = out["pattern"].notna() & np.isfinite(out[["oldest", "middle", "newest"]]).all(axis=1)
        out["pattern"] = out["pattern"].where(valid)
        for k in range(6):
            out[f"p{k}"] = (
                out["pattern"].eq(k).astype(float).where(valid)
                .rolling(params.pattern_window, min_periods=params.pattern_window).mean()
            )
        probabilities = out[[f"p{k}" for k in range(6)]]
        out["complexity"] = (1.2 * (1.0 - probabilities.pow(2).sum(axis=1, min_count=6))).clip(0.0, 1.0)
        out["bias"] = out["p0"] + out["p2"] - out["p4"] - out["p5"]
        out["season"] = (data.index.day >= 25) | (data.index.day <= 10)
        out["strength"] = ((1.0 - out["complexity"]) * out["bias"]).clip(0.0, 1.0).fillna(0.0)
        out["entry"] = (
            out["season"]
            & out["complexity"].lt(params.complexity_cap)
            & out["bias"].gt(params.min_bias)
        ).fillna(False)
        out["target_size"] = (0.15 + 0.80 * out["strength"]).clip(0.15, 0.95)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalSeasonParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # Exposure ends exactly when the same calendar-pattern entry condition flips false.
        df["signal"] = indicators["entry"].astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["target_size"].shift(1).fillna(0.15).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.bdate_range("2020-01-01", periods=100)
    close = np.arange(100, dtype=float) + 100.0
    data = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1000.0}, index=index)
    strategy = GeneratedStrategy()
    params = OrdinalSeasonParams()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    indicators = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    expected = indicators["entry"].astype(int).shift(1).fillna(0).astype(int)
    assert signals["signal"].equals(expected)
    assert signals["signal"].sum() > 0
    assert signals["signal"].isin([0, 1]).all()
    assert np.isfinite(signals["size"]).all() and signals["size"].between(0.15, 0.95).all()
    data["close"] = 100.0
    flat = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert flat["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
