from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReturnImbalanceParams:
    lookback: int = 12
    depression: float = 0.25
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[ReturnImbalanceParams]):
    strategy_id = "gen_local-w2_1790968681"

    @classmethod
    def params_type(cls):
        return ReturnImbalanceParams

    @staticmethod
    def warmup_bars(params: ReturnImbalanceParams) -> int:
        return int(params.lookback + 3)

    @staticmethod
    def indicators(data: pd.DataFrame, params: ReturnImbalanceParams) -> pd.DataFrame:
        if params.lookback < 2 or not 0.0 < params.depression < 1.0:
            raise ValueError("Require lookback >= 2 and depression between zero and one")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between three and five")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["net_return"] = out["return"].rolling(params.lookback, min_periods=params.lookback).sum()
        out["absolute_return"] = out["return"].abs().rolling(params.lookback, min_periods=params.lookback).sum()
        out["imbalance"] = out["net_return"].div(out["absolute_return"].where(out["absolute_return"] > 0.0))
        out["imbalance_change"] = out["imbalance"].diff()
        out["qualifies"] = (
            out["return"].gt(0.0)
            & out["imbalance"].lt(-params.depression)
            & out["imbalance_change"].gt(0.0)
        ).fillna(False)
        out["confirmed"] = out["qualifies"] & out["qualifies"].shift(1, fill_value=False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ReturnImbalanceParams) -> SignalFrame:
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between three and five")
        entries = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                else:
                    signals[i] = 1
            elif entries[i]:
                signals[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame(index=index)
    for confirmations in ([True] * 9, [True] + [False] * 8):
        indicators = pd.DataFrame({"confirmed": confirmations}, index=index)
        result = GeneratedStrategy.generate_signals(data, indicators, None, ReturnImbalanceParams())
        expected = [0, 1, 1, 1, 1, 0, 1, 1, 1] if confirmations[1] else [0, 1, 1, 1, 1, 0, 0, 0, 0]
        assert result.data["signal"].tolist() == expected
