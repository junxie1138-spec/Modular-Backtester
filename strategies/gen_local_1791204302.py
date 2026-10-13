from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    peak_window: int = 20
    change_window: int = 5
    depth_threshold: float = 0.012
    change_threshold: float = 0.003


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = "gen_local_1791204302"

    @classmethod
    def params_type(cls) -> type[DrawdownParams]:
        return DrawdownParams

    def warmup_bars(self, params: DrawdownParams) -> int:
        return int(params.peak_window + params.change_window + 1)

    def indicators(self, data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        if params.peak_window < 2 or params.change_window < 1:
            raise ValueError("Lookbacks must be positive; peak_window must exceed one.")
        if not 0 < params.depth_threshold < 1 or not 0 < params.change_threshold < 1:
            raise ValueError("Depth and change thresholds must lie between zero and one.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(np.isfinite(data["close"]) & data["close"].gt(0))
        out["peak"] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out["drawdown"] = 1.0 - close / out["peak"]
        out["depth_change"] = out["drawdown"].diff(params.change_window)
        out["long_condition"] = (
            out["drawdown"].le(params.depth_threshold)
            & out["depth_change"].le(-params.change_threshold)
        ).fillna(False)
        out["short_condition"] = (
            out["drawdown"].gt(params.depth_threshold)
            & out["depth_change"].ge(params.change_threshold)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownParams,
    ) -> SignalFrame:
        longs = indicators["long_condition"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_condition"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i in range(len(data)):
            if position == 1 and shorts[i]:
                position = 0
            elif position == -1 and longs[i]:
                position = 0
            elif position == 0 and i > 0:
                if longs[i] and longs[i - 1]:
                    position = 1
                elif shorts[i] and shorts[i - 1]:
                    position = -1
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = DrawdownParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    conditions = pd.DataFrame({
        "long_condition": [True, False, True, True, False, False, False, False],
        "short_condition": [False, False, False, False, True, True, False, False],
    }, index=index)
    result = strategy.generate_signals(data, conditions, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 1, 0, -1, -1]
    assert result["size"].gt(0).all()
    warmup = strategy.indicators(data, params)
    assert not warmup[["long_condition", "short_condition"]].to_numpy().any()
