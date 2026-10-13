from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    lookback: int = 21
    smoothing: int = 3
    threshold: float = 0.35


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1790874058"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(params.lookback + params.smoothing)

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if params.lookback < 6 or params.smoothing < 1 or not 0 < params.threshold <= 1:
            raise ValueError("Require lookback >= 6, smoothing >= 1 and 0 < threshold <= 1.")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        prices = out["log_close"]
        maximum = pd.Series(-np.inf, index=data.index)
        signed = pd.Series(0.0, index=data.index)
        total = pd.Series(0.0, index=data.index)
        # A past node is visible when every intervening node lies below its chord.
        # Only long edges contribute; short edges still obstruct visibility.
        for lag in range(1, params.lookback + 1):
            slope = (prices.shift(lag) - prices) / lag
            visible = slope.notna() & slope.gt(maximum)
            if lag >= max(2, params.lookback // 3):
                weight = float(lag * lag)
                total += visible.astype(float) * weight
                signed += visible.astype(float) * (-np.sign(slope)).fillna(0.0) * weight
            maximum = pd.Series(
                np.maximum(maximum.to_numpy(), slope.fillna(-np.inf).to_numpy()),
                index=data.index,
            )
        complete = prices.notna().rolling(
            params.lookback + 1, min_periods=params.lookback + 1
        ).sum().eq(params.lookback + 1)
        out["signed_reach"] = signed.where(complete)
        out["total_reach"] = total.where(complete)
        out["raw_coherence"] = (signed / total.replace(0.0, np.nan)).fillna(0.0).where(complete)
        out["coherence"] = out["raw_coherence"].rolling(
            params.smoothing, min_periods=params.smoothing
        ).mean()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        score = indicators["coherence"]
        votes = np.select(
            [score.ge(params.threshold), score.le(-params.threshold)],
            [1, -1],
            default=0,
        ).astype(int)
        positions = np.zeros(len(data), dtype=int)
        position = previous = 0
        for i, vote in enumerate(votes):
            if position != 0 and vote == -position:
                position = 0
            elif position == 0 and vote != 0 and vote == previous:
                position = int(vote)
            positions[i] = position
            previous = int(vote)
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = VisibilityParams()
    index = pd.date_range("2020-01-01", periods=80)
    for direction in (1, -1):
        t = np.arange(80, dtype=float)
        log_price = 4.0 + 0.0001 * (t + 1.0) ** 2 if direction == 1 else 4.0 + 0.0001 * (100.0 - t) ** 2
        close = np.exp(log_price)
        data = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1000.0},
            index=index,
        )
        indicators = strategy.indicators(data, params)
        assert indicators["coherence"].iloc[:params.lookback + params.smoothing - 1].isna().all()
        assert np.isclose(indicators["coherence"].iloc[-1], direction)
    scores = [np.nan, 0.8, 0.8, 0.0, -0.8, -0.8, 0.0, 0.8, 0.8]
    data = pd.DataFrame(index=index[:len(scores)])
    indicators = pd.DataFrame({"coherence": scores}, index=data.index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
