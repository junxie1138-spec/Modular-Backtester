from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapBreadthParams:
    gap_window: int = 4
    breadth_threshold: float = 2.2
    dominance_threshold: float = 0.6


class GeneratedStrategy(BaseStrategy[GapBreadthParams]):
    strategy_id = "gen_local_1791272746"

    @classmethod
    def params_type(cls) -> type[GapBreadthParams]:
        return GapBreadthParams

    @staticmethod
    def warmup_bars(params: GapBreadthParams) -> int:
        return max(200, params.gap_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapBreadthParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda s: s > 0)
        out["gap"] = (data["open"] / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
        out["positive_gap"] = out["gap"].clip(lower=0.0)
        out["positive_mass"] = out["positive_gap"].rolling(params.gap_window, min_periods=params.gap_window).sum()
        out["positive_energy"] = out["positive_gap"].pow(2).rolling(params.gap_window, min_periods=params.gap_window).sum()
        out["effective_contributors"] = out["positive_mass"].pow(2) / out["positive_energy"].replace(0.0, np.nan)
        out["signed_mass"] = out["gap"].rolling(params.gap_window, min_periods=params.gap_window).sum()
        out["absolute_mass"] = out["gap"].abs().rolling(params.gap_window, min_periods=params.gap_window).sum()
        out["gap_dominance"] = out["signed_mass"] / out["absolute_mass"].replace(0.0, np.nan)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapBreadthParams,
    ) -> SignalFrame:
        condition = (
            (data["close"] > indicators["ma200"])
            & (indicators["effective_contributors"] >= params.breadth_threshold)
            & (indicators["gap_dominance"] >= params.dominance_threshold)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Exposure ends only when the same entry condition becomes false.
        df["signal"] = condition.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = GapBreadthParams()
    gaps = np.concatenate((np.zeros(200), np.full(12, 0.002), np.full(8, -0.004)))
    close = 100.0 * np.cumprod(1.0 + gaps)
    data = pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1000000.0},
        index=pd.date_range("2020-01-01", periods=len(close), freq="B"),
    )
    features = GeneratedStrategy.indicators(data, params)
    signals = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert signals["signal"].iloc[:203].eq(0).all()
    assert signals["signal"].iloc[203:213].eq(1).all()
    assert signals["signal"].iloc[213:].eq(0).all()
    assert signals["size"].gt(0).all()
    assert GeneratedStrategy.warmup_bars(params) >= 200
