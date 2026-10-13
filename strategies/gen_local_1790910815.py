from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapPhaseParams:
    window: int = 6
    threshold: float = 1.0


class GeneratedStrategy(BaseStrategy[GapPhaseParams]):
    strategy_id = "gen_local_1790910815"

    @classmethod
    def params_type(cls):
        return GapPhaseParams

    def warmup_bars(self, params: GapPhaseParams) -> int:
        if not 3 <= params.window <= 6:
            raise ValueError("window must be between 3 and 6")
        if not np.isfinite(params.threshold) or params.threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        return int(params.window + 4)

    def indicators(self, data: pd.DataFrame, params: GapPhaseParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1).where(lambda x: x > 0)
        out["gap"] = (
            data["open"].where(lambda x: x > 0) / out["prior_close"] - 1.0
        ).replace([np.inf, -np.inf], np.nan)
        out["lagged_gap"] = out["gap"].shift(1)
        out["gap_pair"] = out["gap"] + out["lagged_gap"]
        out["lagged_pair"] = out["gap_pair"].shift(2)
        rolling = out["gap_pair"].rolling(params.window, min_periods=params.window)
        out["pair_mean"] = rolling.mean()
        out["pair_scale"] = rolling.std(ddof=0)
        out["pair_correlation"] = rolling.corr(out["lagged_pair"]).clip(-1.0, 1.0)
        out["forecast"] = out["pair_mean"] + out["pair_correlation"] * (
            out["gap_pair"] - out["pair_mean"]
        )
        out["score"] = out["forecast"] / out["pair_scale"].where(
            out["pair_scale"] > 1e-12
        )
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapPhaseParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        scores = indicators["score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, score in enumerate(scores.to_numpy()):
            candidate = (
                1 if score >= params.threshold
                else -1 if score <= -params.threshold
                else 0
            )
            if position == 0:
                position = candidate
            elif candidate == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GapPhaseParams()
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame(
        {"score": [0.0, 2.0, 0.0, -2.0, -2.0, 0.0, 2.0, 2.0]}, index=index
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=10)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) == 10
    assert np.isfinite(strategy.indicators(data, params).to_numpy()).all()


if __name__ == "__main__":
    _self_check()
