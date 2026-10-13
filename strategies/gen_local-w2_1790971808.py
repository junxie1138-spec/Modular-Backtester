from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    lookback: int = 6
    topology_threshold: float = 0.25
    closing_threshold: float = 0.25


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local-w2_1790971808"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(params.lookback) + 1

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if params.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if not 0.0 < params.topology_threshold < 1.0:
            raise ValueError("topology_threshold must be between 0 and 1")
        if not 0.0 < params.closing_threshold < 1.0:
            raise ValueError("closing_threshold must be between 0 and 1")

        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        upper = pd.Series(0.0, index=data.index)
        lower = pd.Series(0.0, index=data.index)
        minimum = pd.Series(np.inf, index=data.index)
        maximum = pd.Series(-np.inf, index=data.index)
        out = pd.DataFrame(index=data.index)

        # A past endpoint is upper-visible when its slope is a new minimum;
        # reflection makes lower visibility a new maximum. Ties are blocked.
        for lag in range(1, params.lookback + 1):
            slope = (close - close.shift(lag)) / float(lag)
            out[f"slope_{lag}"] = slope
            upper += float(lag) * slope.lt(minimum).astype(float)
            lower += float(lag) * slope.gt(maximum).astype(float)
            minimum = pd.Series(np.minimum(minimum.to_numpy(), slope.to_numpy()), index=data.index)
            maximum = pd.Series(np.maximum(maximum.to_numpy(), slope.to_numpy()), index=data.index)

        valid = close.rolling(params.lookback + 1, min_periods=params.lookback + 1).count().eq(params.lookback + 1)
        normalizer = float(params.lookback * (params.lookback + 1) // 2)
        out["upper_reach"] = upper.where(valid)
        out["lower_reach"] = lower.where(valid)
        out["topology"] = ((lower - upper) / normalizer).where(valid)

        high = data["high"].astype(float).replace([np.inf, -np.inf], np.nan)
        low = data["low"].astype(float).replace([np.inf, -np.inf], np.nan)
        span = (high - low).where(high > low)
        out["closing_position"] = ((2.0 * close - high - low) / span).clip(-1.0, 1.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        topology = indicators["topology"]
        position = indicators["closing_position"]
        long_entry = topology.ge(params.topology_threshold) & position.ge(params.closing_threshold)
        short_entry = topology.le(-params.topology_threshold) & position.le(-params.closing_threshold)
        direction = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)

        signal = np.zeros(len(data), dtype=int)
        state = 0
        for bar, candidate in enumerate(direction):
            if state == 0:
                state = int(candidate)
            elif candidate == -state:
                # Opposite agreement closes the position; neutral bars hold.
                state = 0
            signal[bar] = state

        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = VisibilityParams()
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    data = pd.DataFrame({"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 100.0}, index=index)
    values = [1.0, np.nan, -1.0, -1.0, 0.0, 1.0, 0.0]
    indicators = pd.DataFrame({"topology": values, "closing_position": values}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0.0).all()
    flat = GeneratedStrategy.indicators(None, data, params)
    assert flat["topology"].iloc[:params.lookback].isna().all()
    assert flat["topology"].iloc[-1] == 0.0


if __name__ == "__main__":
    _self_check()
