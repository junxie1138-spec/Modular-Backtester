from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RankShiftParams:
    rank_window: int = 48
    shift_threshold: float = 0.22


class GeneratedStrategy(BaseStrategy[RankShiftParams]):
    strategy_id = "gen_local-w2_1791325079"

    @classmethod
    def params_type(cls) -> type[RankShiftParams]:
        return RankShiftParams

    def warmup_bars(self, params: RankShiftParams) -> int:
        return int(params.rank_window) + 8

    def indicators(self, data: pd.DataFrame, params: RankShiftParams) -> pd.DataFrame:
        if params.rank_window < 8:
            raise ValueError("rank_window must be at least 8")
        if not np.isfinite(params.shift_threshold) or not 0.0 < params.shift_threshold < 1.0:
            raise ValueError("shift_threshold must be between zero and one")
        close = data["close"].where(data["close"] > 0.0)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        percentile = returns.rolling(params.rank_window, min_periods=params.rank_window).rank(method="average", pct=True)
        recent = percentile.rolling(4, min_periods=4).mean()
        preceding = recent.shift(4)
        return pd.DataFrame({
            "return": returns,
            "return_percentile": percentile,
            "recent_rank_mean": recent,
            "preceding_rank_mean": preceding,
            "rank_shift": recent - preceding,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RankShiftParams) -> SignalFrame:
        score = indicators["rank_shift"]
        events = pd.Series(0, index=data.index, dtype=int)
        events.loc[score.ge(params.shift_threshold)] = 1
        events.loc[score.le(-params.shift_threshold)] = -1
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, event in enumerate(events.to_numpy()):
            if position == 0:
                position = int(event)
            elif event == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = RankShiftParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"rank_shift": [0.3, np.nan, -0.3, -0.3, 0.0, 0.3, 0.3]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0.0).all()
    flat = data.reindex(pd.date_range("2020-01-01", periods=100)).ffill()
    derived = strategy.indicators(flat, params)
    assert derived["rank_shift"].first_valid_index() == flat.index[params.rank_window + 7]
    assert derived["rank_shift"].dropna().eq(0.0).all()
    assert strategy.generate_signals(flat, derived, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
