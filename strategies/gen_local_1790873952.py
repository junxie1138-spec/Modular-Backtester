from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapVoteParams:
    scale_window: int = 15
    vote_window: int = 7
    threshold: float = 0.35


class GeneratedStrategy(BaseStrategy[GapVoteParams]):
    strategy_id = "gen_local_1790873952"

    @classmethod
    def params_type(cls) -> type[GapVoteParams]:
        return GapVoteParams

    @staticmethod
    def warmup_bars(params: GapVoteParams) -> int:
        return int(params.scale_window + params.vote_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapVoteParams) -> pd.DataFrame:
        if params.scale_window < 2 or params.vote_window < 2:
            raise ValueError("Lookback windows must be at least two bars")
        if not 0.0 < params.threshold < 1.0:
            raise ValueError("threshold must lie strictly between zero and one")
        previous_close = data["close"].shift(1)
        gap = (data["open"] / previous_close.where(previous_close > 0) - 1.0)
        gap = gap.replace([np.inf, -np.inf], np.nan)
        scale = gap.abs().rolling(params.scale_window, min_periods=params.scale_window).median().shift(1)
        scale = scale.clip(lower=1e-8)
        vote = np.tanh(gap / scale)
        score = vote.rolling(params.vote_window, min_periods=params.vote_window).mean()
        return pd.DataFrame({"gap": gap, "gap_scale": scale, "gap_vote": vote, "score": score}, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapVoteParams) -> SignalFrame:
        score = indicators["score"]
        state = pd.Series(np.nan, index=data.index, dtype=float)
        state.loc[score >= params.threshold] = 1.0
        state.loc[score <= -params.threshold] = 0.0
        df = pd.DataFrame(index=data.index)
        df["signal"] = state.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = GapVoteParams()
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"score": [np.nan, 0.5, 0.1, -0.1, -0.5, -0.1, 0.5, 0.0]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 1]
    assert result["size"].gt(0).all()
    flat_index = pd.date_range("2020-01-01", periods=40, freq="D")
    flat = data.iloc[:1].reindex(flat_index).fillna(data.iloc[0])
    calculated = strategy.indicators(flat, params)
    assert calculated["score"].first_valid_index() == flat_index[strategy.warmup_bars(params) - 1]
    assert calculated["score"].dropna().eq(0.0).all()


if __name__ == "__main__":
    _self_check()
