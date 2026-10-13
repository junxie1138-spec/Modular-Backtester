from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    history_bars: int = 126
    min_edge: float = 0.0005


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791283846"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(params.history_bars) + 23

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if params.history_bars < 30 or not np.isfinite(params.min_edge) or params.min_edge < 0:
            raise ValueError("history_bars must be >= 30 and min_edge must be finite and nonnegative")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["rank"] = out["return"].rolling(20, min_periods=20).rank(pct=True)
        out["bucket"] = np.floor(out["rank"] * 3).clip(upper=2)
        out["shape"] = out["bucket"] + 3 * out["bucket"].shift(1) + 9 * out["bucket"].shift(2)
        out["open_return"] = data["open"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        count = pd.Series(0.0, index=data.index)
        total = count.copy()
        squares = count.copy()
        # Each analog's next-open to following-open return matures two bars after its shape.
        # ponytail: O(history_bars * bars); cache shape aggregates if histories become large.
        for lag in range(4, int(params.history_bars) + 1):
            outcome = out["open_return"].shift(lag - 2)
            match = out["shape"].eq(out["shape"].shift(lag)) & outcome.notna()
            sample = outcome.where(match, 0.0)
            count += match.astype(float)
            total += sample
            squares += sample * sample
        out["matches"] = count
        denominator = count.where(count >= 5)
        out["forecast"] = total / denominator
        variance = ((squares - total * total / denominator) / (denominator - 1)).clip(lower=0)
        out["standard_error"] = np.sqrt(variance / denominator)
        out["edge"] = out["forecast"] - 0.5 * out["standard_error"]
        out["edge"] = out["edge"].where(np.arange(len(data)) >= self.warmup_bars(params) - 1)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # The same condition enters and maintains a position; its reversal exits.
        df["signal"] = (indicators["edge"].notna() & indicators["edge"].gt(params.min_edge)).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = AnalogParams()
    n = 400
    t = np.arange(n, dtype=float)
    close = 100.0 * np.exp(np.cumsum(0.001 + 0.008 * np.sin(t * 0.7)))
    opening = close * np.exp(0.002 * np.cos(t * 0.4))
    data = pd.DataFrame(
        {"open": opening, "high": np.maximum(opening, close) * 1.01,
         "low": np.minimum(opening, close) * 0.99, "close": close, "volume": 1000000.0},
        index=pd.date_range("2020-01-01", periods=n),
    )
    indicators = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, indicators, None, params).data
    expected = indicators["edge"].gt(params.min_edge).astype(int).shift(1).fillna(0).astype(int)
    assert signals["signal"].equals(expected)
    assert signals["signal"].isin([0, 1]).all() and signals["size"].gt(0).all()
    assert signals["signal"].iloc[:strategy.warmup_bars(params)].eq(0).all()
    prefix = strategy.indicators(data.iloc[:300], params)
    pd.testing.assert_frame_equal(prefix, indicators.iloc[:300])
    probe = pd.DataFrame({"edge": [0.002, -0.001, 0.002, -0.001]}, index=data.index[:4])
    reversal = strategy.generate_signals(data.iloc[:4], probe, None, params).data["signal"]
    assert reversal.tolist() == [0, 1, 0, 1]


if __name__ == "__main__":
    _self_check()
