from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TerritoryParams:
    range_bars: int = 18
    streak_bars: int = 3


class GeneratedStrategy(BaseStrategy[TerritoryParams]):
    strategy_id = "gen_local_1791245869"

    @classmethod
    def params_type(cls) -> type[TerritoryParams]:
        return TerritoryParams

    def warmup_bars(self, params: TerritoryParams) -> int:
        return int(max(200, params.range_bars + 1) + params.streak_bars - 1)

    def indicators(self, data: pd.DataFrame, params: TerritoryParams) -> pd.DataFrame:
        if params.range_bars < 2 or params.streak_bars < 1:
            raise ValueError("range_bars must be >= 2 and streak_bars must be >= 1")
        out = pd.DataFrame(index=data.index)
        out["prior_high"] = data["high"].rolling(params.range_bars, min_periods=params.range_bars).max().shift(1)
        out["prior_low"] = data["low"].rolling(params.range_bars, min_periods=params.range_bars).min().shift(1)
        out["midpoint"] = (out["prior_high"] + out["prior_low"]) / 2.0
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["ready"] = (
            np.isfinite(out[["prior_high", "prior_low", "midpoint", "ma200"]]).all(axis=1)
            & np.isfinite(data[["high", "low", "close"]]).all(axis=1)
            & data["high"].ge(data["low"])
            & out["prior_high"].gt(out["prior_low"])
        )
        out["upper_acceptance"] = out["ready"] & data["low"].gt(out["midpoint"])
        out["lower_acceptance"] = out["ready"] & data["high"].lt(out["midpoint"])
        for side in ("upper", "lower"):
            accepted = out[f"{side}_acceptance"]
            out[f"{side}_streak"] = accepted.astype(int).groupby((~accepted).cumsum()).cumsum().astype(int)
        out["bull_regime"] = out["ready"] & data["close"].gt(out["ma200"])
        out["bear_regime"] = out["ready"] & data["close"].lt(out["ma200"])
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TerritoryParams,
    ) -> SignalFrame:
        upper = indicators["upper_streak"].fillna(0).ge(params.streak_bars)
        lower = indicators["lower_streak"].fillna(0).ge(params.streak_bars)
        ready = indicators["ready"].fillna(False).astype(bool)
        direction = pd.Series(np.select([ready & upper, ready & lower], [1, -1], default=0), index=data.index)
        eligible = (
            (direction.eq(1) & indicators["bull_regime"].fillna(False))
            | (direction.eq(-1) & indicators["bear_regime"].fillna(False))
        )
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, (candidate, can_enter) in enumerate(zip(direction.to_numpy(), eligible.to_numpy())):
            if position != 0:
                if candidate == -position:
                    position = 0
            elif can_enter:
                position = int(candidate)
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "upper_streak": [3, 0, 0, 0, 0, 0, 3, 0, 0],
        "lower_streak": [0, 0, 3, 3, 0, 0, 0, 0, 0],
        "ready": [True] * 9,
        "bull_regime": [True, False, False, False, True, True, True, False, False],
        "bear_regime": [False, True, True, True, False, False, False, True, True],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, TerritoryParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, -1, 0, 0]
    assert result["size"].gt(0).all()
    assert result.index.equals(data.index)


if __name__ == "__main__":
    _self_check()
