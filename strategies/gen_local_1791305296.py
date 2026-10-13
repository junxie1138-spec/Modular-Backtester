from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapTopologyParams:
    window: int = 4
    gap_weight: float = 1.0


class GeneratedStrategy(BaseStrategy[GapTopologyParams]):
    strategy_id = "gen_local_1791305296"

    @classmethod
    def params_type(cls) -> type[GapTopologyParams]:
        return GapTopologyParams

    def warmup_bars(self, params: GapTopologyParams) -> int:
        return int(params.window) + 1

    def indicators(self, data: pd.DataFrame, params: GapTopologyParams) -> pd.DataFrame:
        window = int(params.window)
        if window < 3 or not np.isfinite(params.gap_weight) or params.gap_weight <= 0:
            raise ValueError("window must be >= 3 and gap_weight finite and positive")
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1)
        out["previous_close"] = previous.where(previous > 0)
        out["gap"] = (data["open"] / out["previous_close"] - 1).replace([np.inf, -np.inf], np.nan)
        out["intraday"] = ((data["close"] - data["open"]) / out["previous_close"]).replace([np.inf, -np.inf], np.nan)
        out["gap_debt"] = (params.gap_weight * out["gap"] - out["intraday"]).replace([np.inf, -np.inf], np.nan)
        current = out["gap_debt"].to_numpy(dtype=float)
        upper = np.zeros(len(data), dtype=int)
        lower = np.zeros(len(data), dtype=int)
        upper_blocker = np.full(len(data), -np.inf)
        lower_blocker = np.full(len(data), np.inf)
        # Horizontal visibility: intervening nodes must lie strictly below
        # both endpoints; the lower graph applies the mirrored inequality.
        for lag in range(1, window):
            past = out["gap_debt"].shift(lag).to_numpy(dtype=float)
            valid = np.isfinite(current) & np.isfinite(past)
            upper += valid & (upper_blocker < np.minimum(current, past))
            lower += valid & (lower_blocker > np.maximum(current, past))
            upper_blocker = np.maximum(upper_blocker, past)
            lower_blocker = np.minimum(lower_blocker, past)
        out["upper_degree"] = upper
        out["lower_degree"] = lower
        out["ready"] = out["gap_debt"].rolling(window, min_periods=window).count().eq(window)
        out["score"] = (out["lower_degree"] - out["upper_degree"]).where(out["ready"], 0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapTopologyParams) -> SignalFrame:
        decisions = pd.Series(np.nan, index=data.index, dtype=float)
        # Mirrored rules: positive degree imbalance enters; negative exits.
        # Neutral topology preserves the position until reversal.
        decisions.loc[indicators["score"] > 0] = 1.0
        decisions.loc[indicators["score"] < 0] = 0.0
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = GapTopologyParams(window=3)
    index = pd.date_range("2020-01-01", periods=8)
    gaps = np.array([-0.04, 0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01])
    data = pd.DataFrame({"open": 100 * (1 + gaps), "close": 100.0, "volume": 1000.0}, index=index)
    data["high"] = data[["open", "close"]].max(axis=1) + 1
    data["low"] = data[["open", "close"]].min(axis=1) - 1
    indicators = GeneratedStrategy.indicators(None, data, params)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 1, 0, 1]
    reflected = data.copy()
    reflected["open"] = 200 - data["open"]
    reflected["high"] = 200 - data["low"]
    reflected["low"] = 200 - data["high"]
    opposite = GeneratedStrategy.indicators(None, reflected, params)
    assert np.array_equal(indicators["score"].to_numpy(), -opposite["score"].to_numpy())
    for length in range(1, len(data) + 1):
        prefix = GeneratedStrategy.indicators(None, data.iloc[:length], params)
        assert np.array_equal(prefix["score"].to_numpy(), indicators["score"].iloc[:length].to_numpy())
    assert result["size"].gt(0).all()
