from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ParticipationParams:
    window: int = 8
    crowding: float = 0.25
    dominance: float = 0.15


class GeneratedStrategy(BaseStrategy[ParticipationParams]):
    strategy_id = "gen_local-w2_1791294131"

    @classmethod
    def params_type(cls) -> type[ParticipationParams]:
        return ParticipationParams

    def warmup_bars(self, params: ParticipationParams) -> int:
        if not 2 <= params.window <= 9:
            raise ValueError("window must be between 2 and 9")
        if not 0.0 < params.crowding <= 1.0:
            raise ValueError("crowding must be in (0, 1]")
        if not 0.0 < params.dominance <= 1.0:
            raise ValueError("dominance must be in (0, 1]")
        return int(params.window + 1)

    def indicators(self, data: pd.DataFrame, params: ParticipationParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return_sign"] = np.sign(out["return"])
        out["absolute_return"] = out["return"].abs()
        out["breadth"] = out["return_sign"].rolling(params.window, min_periods=params.window).mean()
        out["net_return"] = out["return"].rolling(params.window, min_periods=params.window).sum()
        out["gross_return"] = out["absolute_return"].rolling(params.window, min_periods=params.window).sum()
        out["dominance"] = out["net_return"] / out["gross_return"].replace(0.0, np.nan)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ParticipationParams) -> SignalFrame:
        long_entry = (indicators["breadth"] <= -params.crowding) & (indicators["dominance"] >= params.dominance)
        short_entry = (indicators["breadth"] >= params.crowding) & (indicators["dominance"] <= -params.dominance)
        votes = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, vote in enumerate(votes):
            if position == 0:
                position = int(vote)
            elif vote == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = ParticipationParams()
    index = pd.date_range("2020-01-01", periods=9)
    returns = np.array([0.0] + [-0.001] * 6 + [0.015, -0.001])
    close = 100.0 * np.cumprod(1.0 + returns)
    data = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1000.0}, index=index)
    actual = strategy.indicators(data, params)
    assert strategy.warmup_bars(params) == 9
    assert actual["breadth"].iloc[:8].isna().all()
    assert actual["breadth"].iloc[-1] <= -params.crowding
    assert actual["dominance"].iloc[-1] >= params.dominance
    votes = np.array([1, 1, -1, -1, 0, 1, 1])
    evidence = pd.DataFrame({"breadth": -0.5 * votes, "dominance": 0.5 * votes}, index=index[:7])
    result = strategy.generate_signals(data.iloc[:7], evidence, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert (result["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()
