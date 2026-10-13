from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    rank_window: int = 63
    impulse_bars: int = 4
    impact_bars: int = 5
    regime_cutoff: float = 0.5
    entry_tail: float = 0.35
    min_size: float = 0.2
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790934366"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(params.rank_window + max(params.impulse_bars, params.impact_bars) + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if min(params.rank_window, params.impulse_bars, params.impact_bars) < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not (0 < params.regime_cutoff < 1 and 0 < params.entry_tail < 0.5):
            raise ValueError("Invalid percentile thresholds.")
        if not (0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["daily_return"] = close.pct_change(fill_method=None)
        out["impulse"] = close.pct_change(params.impulse_bars, fill_method=None)
        out["dollar_volume"] = close * data["volume"].where(data["volume"] > 0)
        out["price_impact"] = out["daily_return"].abs() / out["dollar_volume"]
        out["average_impact"] = out["price_impact"].rolling(params.impact_bars).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["impact_rank"] = out["average_impact"].rolling(params.rank_window).rank(pct=True)
        out["impulse_rank"] = out["impulse"].rolling(params.rank_window).rank(pct=True)
        out["transmission_regime"] = out["impact_rank"] > params.regime_cutoff
        out["signal_strength"] = pd.Series(
            np.where(
                out["transmission_regime"],
                (out["impulse_rank"] - (1 - params.entry_tail)) / params.entry_tail,
                (params.entry_tail - out["impulse_rank"]) / params.entry_tail,
            ),
            index=data.index,
        ).clip(0, 1).fillna(0.0)
        ready = out[["impact_rank", "impulse_rank", "impulse"]].notna().all(axis=1)
        direction = (
            (out["transmission_regime"] & out["impulse"].gt(0))
            | (~out["transmission_regime"] & out["impulse"].lt(0))
        )
        out["long_condition"] = ready & direction & out["signal_strength"].gt(0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # The target remains long only while the entry predicate holds.
        # Its first false reading is the sole exit rule.
        df["signal"] = indicators["long_condition"].fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        strength = indicators["signal_strength"].fillna(0.0).clip(0, 1)
        df["size"] = (
            params.min_size + (params.max_size - params.min_size) * strength
        ).shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GeneratedParams(rank_window=21)
    t = np.arange(200, dtype=float)
    close = 100 * np.exp(np.cumsum(0.01 * np.sin(t / 3) + 0.002 * np.cos(t / 11)))
    data = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1e6 * (1 + 0.4 * np.sin(t / 9))},
        index=pd.date_range("2020-01-01", periods=len(t)),
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    derived = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, derived, ctx, params).data
    expected = derived["long_condition"].shift(1).fillna(False).astype(int)
    assert signals["signal"].equals(expected)
    assert signals["signal"].eq(1).any() and signals["signal"].diff().eq(-1).any()
    assert signals["size"].between(params.min_size, params.max_size).all()
    assert signals["size"].nunique() > 2
    prefix = data.iloc[:150]
    prefix_signals = strategy.generate_signals(
        prefix, strategy.indicators(prefix, params), ctx, params
    ).data
    pd.testing.assert_frame_equal(signals.iloc[:150], prefix_signals)
