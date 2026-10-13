from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RepairParams:
    peak_window: int = 20
    rank_window: int = 5
    tail: float = 0.7


class GeneratedStrategy(BaseStrategy[RepairParams]):
    strategy_id = "gen_local_1790897242"

    @classmethod
    def params_type(cls):
        return RepairParams

    def warmup_bars(self, params: RepairParams) -> int:
        return int(params.peak_window + params.rank_window + 2)

    def indicators(self, data: pd.DataFrame, params: RepairParams) -> pd.DataFrame:
        if params.peak_window < 2 or params.rank_window < 3 or not 0.5 < params.tail < 1.0:
            raise ValueError("Windows must be >=2 and >=3; tail must be between 0.5 and 1.")
        out = pd.DataFrame(index=data.index)
        out["peak"] = data["high"].rolling(params.peak_window).max()
        out["mean_volume"] = data["volume"].rolling(params.peak_window).mean()
        out["relative_volume"] = data["volume"].where(data["volume"] > 0).div(
            out["mean_volume"].where(out["mean_volume"] > 0)
        )
        out["drawdown"] = data["close"].div(out["peak"].where(out["peak"] > 0)) - 1.0
        out["repair"] = data["close"].diff().div(
            out["peak"].shift(1).where(out["peak"].shift(1) > 0)
        ).div(out["relative_volume"])
        out["repair_rank"] = out["repair"].rolling(params.rank_window).rank(pct=True)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RepairParams,
    ) -> SignalFrame:
        repair = indicators["repair"]
        rank = indicators["repair_rank"]
        direction = pd.Series(0, index=data.index, dtype=int)
        direction.loc[repair.gt(0) & rank.ge(params.tail)] = 1
        direction.loc[repair.lt(0) & rank.le(1.0 - params.tail)] = -1
        entry = direction.where(indicators["drawdown"].lt(0), 0)
        confirmed = entry.ne(0) & entry.eq(entry.shift(1))
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i in range(len(data)):
            candidate = int(direction.iat[i])
            # Opposite repair polarity exits immediately; neutral bars keep the position.
            if position and candidate == -position:
                position = 0
            elif position == 0 and bool(confirmed.iat[i]):
                position = int(entry.iat[i])
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "drawdown": [-0.1] * 8,
        "repair": [1.0, 1.0, np.nan, -1.0, -1.0, 0.0, 1.0, 1.0],
        "repair_rank": [0.9, 0.9, np.nan, 0.1, 0.1, 0.5, 0.9, 0.9],
    }, index=index)
    strategy = GeneratedStrategy()
    params = RepairParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
