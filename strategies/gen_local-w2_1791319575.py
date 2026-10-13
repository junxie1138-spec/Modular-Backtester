from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    window: int = 12
    min_degree: int = 3


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local-w2_1791319575"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return max(4, int(params.window)) + 1

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        window = max(4, int(params.window))
        close = data["close"].astype(float).where(data["close"] > 0)
        price = np.log(close)
        upper = np.full(len(data), -np.inf)
        lower = np.full(len(data), np.inf)
        degree = np.zeros(len(data), dtype=int)
        inverse_degree = np.zeros(len(data), dtype=int)
        current = price.to_numpy()
        # Horizontal visibility: every intervening node must lie below both endpoints.
        for lag in range(1, window):
            past = price.shift(lag).to_numpy()
            degree += (current > upper) & (past > upper)
            inverse_degree += (current < lower) & (past < lower)
            upper = np.maximum(upper, past)
            lower = np.minimum(lower, past)
        ready = price.rolling(window, min_periods=window).count().eq(window)
        out = pd.DataFrame(index=data.index)
        out["log_close"] = price
        out["degree"] = pd.Series(degree, index=data.index).where(ready)
        out["inverse_degree"] = pd.Series(inverse_degree, index=data.index).where(ready)
        out["peak"] = close.rolling(window, min_periods=window).max()
        out["drawdown"] = close / out["peak"] - 1.0
        out["change"] = price.diff()
        threshold = max(2, int(params.min_degree))
        recovery = (
            (out["drawdown"] < 0)
            & (out["change"] > 0)
            & (out["degree"] >= threshold)
            & (out["degree"] > out["degree"].shift(1))
        )
        failed_recovery = (
            (out["drawdown"] < 0)
            & (out["change"] < 0)
            & (out["change"].shift(1) > 0)
            & (out["inverse_degree"] >= threshold)
            & (out["inverse_degree"] > out["inverse_degree"].shift(1))
        )
        out["entry"] = np.select([recovery, failed_recovery], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        entries = indicators["entry"].reindex(data.index).fillna(0).to_numpy(dtype=int)
        positions = np.zeros(len(data), dtype=int)
        side = 0
        exit_bar = -1
        for bar, entry in enumerate(entries):
            if side and bar == exit_bar:
                side = 0
                continue
            if not side and entry:
                side = int(np.sign(entry))
                exit_bar = bar + 2
            positions[bar] = side
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = VisibilityParams(window=4, min_degree=2)
    index = pd.date_range("2020-01-01", periods=9)
    close = pd.Series([10, 8, 6, 7, 6, 8, 7, 9, 8], index=index, dtype=float)
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1000.0}, index=index)
    features = strategy.indicators(data, params)
    assert features["degree"].iloc[3] == 2
    assert features["inverse_degree"].iloc[4] == 2
    scripted = pd.DataFrame({"entry": [1, -1, 1, -1, 1, 1, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, scripted, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 0, 0]
    prefix = strategy.indicators(data.iloc[:7], params)
    pd.testing.assert_frame_equal(features.iloc[:7], prefix)
