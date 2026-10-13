from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SessionInversionParams:
    lookback: int = 40
    tail_fraction: float = 0.20

    def __post_init__(self):
        if self.lookback < 4 or not 0.0 < self.tail_fraction < 0.5:
            raise ValueError("Require lookback >= 4 and 0 < tail_fraction < 0.5")


class GeneratedStrategy(BaseStrategy[SessionInversionParams]):
    strategy_id = "gen_local_1790878938"

    @classmethod
    def params_type(cls):
        return SessionInversionParams

    def warmup_bars(self, params: SessionInversionParams) -> int:
        return params.lookback + 2

    def indicators(self, data: pd.DataFrame, params: SessionInversionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        out["gap"] = np.log(opening / close.shift(1)).replace([np.inf, -np.inf], np.nan)
        out["intraday"] = np.log(close / opening).replace([np.inf, -np.inf], np.nan)
        out["leadership"] = out["intraday"] - out["gap"]
        out["leadership_rank"] = out["leadership"].rolling(
            params.lookback, min_periods=params.lookback
        ).rank(pct=True)
        out["previous_rank"] = out["leadership_rank"].shift(1)
        out["entry"] = (
            (out["previous_rank"] <= params.tail_fraction)
            & (out["leadership_rank"] >= 1.0 - params.tail_fraction)
            & (out["gap"] < 0.0)
            & (out["intraday"] > 0.0)
            & (close >= close.shift(1))
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SessionInversionParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, entry in enumerate(entries):
            if i < exit_bar:
                signal[i] = 1
            elif i == exit_bar:
                signal[i] = 0
            elif entry:
                signal[i] = 1
                exit_bar = i + 2
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = SessionInversionParams()
    sample = pd.DataFrame(
        {name: np.full(7, 100.0) for name in ("open", "high", "low", "close", "volume")},
        index=pd.date_range("2020-01-01", periods=7),
    )
    derived = strategy.indicators(sample, params)
    assert not derived["entry"].any()
    derived["entry"] = True
    context = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(sample, derived, context, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 1, 1, 0]
    assert (result["size"] > 0).all()
